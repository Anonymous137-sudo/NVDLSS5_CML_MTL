import AppKit
import CoreVideo
import Foundation
import Metal
import MLX

struct NativeFlowPair {
 let from:Double
 let to:Double
 let forward:(CVPixelBuffer,MTLTexture)
 let backward:(CVPixelBuffer,MTLTexture)
}

struct AppearanceProxy {
 let source: (CVPixelBuffer, MTLTexture)
 let correction: MTLTexture
 let colorControls: SIMD3<Float>
 let gain: Float
 let hud: SIMD4<Float>
}

struct CorrectionEvidence {
 let model:Data
 let before:Data
 let after:Data
 let width:Int
 let height:Int
 let rowBytes:Int
}
public final class ProcessedFrame {
 var correctionEvidence: CorrectionEvidence?
 public let pixels:CVPixelBuffer,texture:MTLTexture,sequence:UInt64,sourceSeconds:Double
 public let processingMS:Double,gpuMS:Double,aneMS:Double
 public var presentationID:UInt64=0
 public var generated=false,phase:Double=1
 public var pairStart:UInt64=0,pairEnd:UInt64=0
 public var validation=false
 var motionGuide:CaptureGuide?
 var appearanceProxy: AppearanceProxy?
 var nativeFlow: NativeFlowPair?
 public var captureGuided=false
 public var sourcePixels:CVPixelBuffer?
 public var stages:[String:Double]=[:]
 public var variants:[String:CVPixelBuffer]=[:]
 public let neural=true,appearance=true
 init(_ p:CVPixelBuffer,_ t:MTLTexture,_ seq:UInt64,_ pts:Double,_ ms:Double,_ gpu:Double,_ ane:Double=0) {pixels=p;texture=t;sequence=seq;sourceSeconds=pts;processingMS=ms;gpuMS=gpu;aneMS=ane}
}
public struct NRStats {
 public var submitted:UInt64=0,superseded:UInt64=0,completed:UInt64=0,neuralCompleted:UInt64=0,appearanceCompleted:UInt64=0,failed:UInt64=0
 public var error="",busy=false
}
/// All source instances share one NR executor. Each stage retains one newest
/// pending frame; MFG never blocks NR submission or the presentation thread.
public final class PostPipeline {
 private static let worker=DispatchQueue(label:"obs.dlss-nr.mlx",qos:.userInitiated)
 private let mfgWorker=DispatchQueue(label:"obs.dlss-nr.mfg",qos:.userInitiated)
 private struct MFGWork {
  let raw:CVPixelBuffer,result:ProcessedFrame,config:NRConfiguration,epoch:UInt64
  let began:Double,nrReady:Double
 }
 private var pendingMFG:MFGWork?
 private var mfgBusy=false
 private var mfgSuperseded:UInt64=0,published:UInt64=0
 private var cachedNativeState=("inactive", "")
 private var cachedNativeSkipped:UInt64=0
 public let device:MTLDevice
 public let experimentalAppearance=false
 private let lock=NSLock()
 private var configuration:NRConfiguration
 private var engine:UpstreamNREngine?,engineConfiguration:NRConfiguration?
 private var engineEpoch:UInt64=UInt64.max
 private let guideRing:CaptureGuideRing
 private let pairEvidence=LiveFrameDiagnostics()
 private var pending:(CVPixelBuffer,UInt64,Double,UInt64,CaptureGuide?)?
 private var epoch:UInt64=0,stopped=false
 private var frame:ProcessedFrame?,state=NRStats()
 private var history:[[String:Any]]=[]
 private var appleInterpolator:AppleFGService?,appleEpoch:UInt64=UInt64.max
 private var rifeInterpolator:RifeFrameInterpolator?,rifeEpoch:UInt64=UInt64.max
 private var interpolator:MultiFrameInterpolator?,interpolatorPath="",interpolatorEpoch:UInt64=UInt64.max
 private var schedule=FrameSchedule<ProcessedFrame>(),nextPresentationID:UInt64=0
 private var guidedCount:UInt64=0
 private var generatedCount:UInt64=0,interpolationFailures:UInt64=0,pairResets:UInt64=0
 private var interpolationError=""
 private var warpParity = -1
 private var nativePresentation:[String:Any]=[:]
 private var overlayEnabled=false
 public func setStatusOverlay(_ enabled:Bool){lock.lock();overlayEnabled=enabled;lock.unlock()}
 public var statusOverlayText:String? {lock.lock();defer{lock.unlock()};return overlayEnabled ? (configuration.appliesNR ? "DLSS ON" : configuration.frameGenerationFactor>1 ? "MFG ONLY" : "DLSS OFF") : nil}
 public func reportPresentation(_ report:[String:Any]) {lock.lock();nativePresentation=report;lock.unlock()}

 public var modeLabel:String {lock.lock();defer{lock.unlock()};return configuration.title+" ·native NR \(configuration.nativeProcessingHeight ?? 1080)p ·"+configuration.frameGenerationBackend+" synthetic"+" ·MFG≤\(configuration.frameGenerationFactor)x"}
 public var presentationGeneration:UInt64 {lock.lock();defer{lock.unlock()};return epoch}
 public var stats:NRStats {lock.lock();defer{lock.unlock()};return state}
 public func latest()->ProcessedFrame? {presentationFrame(at:ProcessInfo.processInfo.systemUptime)}
 public func presentationFrame(at target:Double)->ProcessedFrame? {
  lock.lock();defer{lock.unlock()}
  guard configuration.frameGenerationFactor>1 else {return frame}
  let now=ProcessInfo.processInfo.systemUptime
  _ = schedule.sample(at:now)
  return schedule.peek(at:min(now+1.0/30,max(now,target)))
 }
 package init(_ config:NRConfiguration)throws {
  guard let device=MTLCreateSystemDefaultDevice() else {throw NSError(domain:"DLSS NR",code:2)}
  self.device=device;guideRing=CaptureGuideRing(device:device);configuration=try config.checked()
 }
 package func update(_ config:NRConfiguration) {
  lock.lock();defer{lock.unlock()}
  guard config != configuration else {return}
  configuration=config;epoch &+= 1;guideRing.clear(epoch:epoch);pending=nil;pendingMFG=nil;frame=nil;history.removeAll();schedule.clear();state.error="";interpolationError="";warpParity = -1
 }
 public func resetInput() {
  lock.lock();defer{lock.unlock()}
  epoch &+= 1;guideRing.clear(epoch:epoch);pending=nil;pendingMFG=nil;frame=nil;history.removeAll();schedule.clear()
  state.error="";interpolationError=""
 }
 public func submit(_ pixels:CVPixelBuffer,sequence:UInt64,sourceSeconds:Double) {
  lock.lock();let copyGuides=configuration.appliesNR && configuration.frameGenerationFactor>1 && configuration.frameGenerationBackend != "rife";let generation=epoch;let active = !stopped;lock.unlock()
  guard active else{return}
  let guide=copyGuides ? guideRing.copy(pixels,sequence:sequence,seconds:sourceSeconds,epoch:generation) : nil
  lock.lock()
  guard !stopped && generation==epoch else {lock.unlock();return}
  state.submitted &+= 1
  if pending != nil {state.superseded &+= 1}
  pending=(pixels,sequence,sourceSeconds,epoch,guide)
  let start = !state.busy
  if start {state.busy=true}
  lock.unlock()
  if start {Self.worker.async {[self] in drain()}}
 }
 private func rawFrame(_ input:CVPixelBuffer,sequence:UInt64,pts:Double)throws->ProcessedFrame {
  let start=ProcessInfo.processInfo.systemUptime
  let width=CVPixelBufferGetWidth(input),height=CVPixelBufferGetHeight(input)
  let attrs=[kCVPixelBufferIOSurfacePropertiesKey:[:] as CFDictionary] as CFDictionary
  var output:CVPixelBuffer?
  guard CVPixelBufferGetPixelFormatType(input)==kCVPixelFormatType_32BGRA,
        CVPixelBufferCreate(nil,width,height,kCVPixelFormatType_32BGRA,attrs,&output)==kCVReturnSuccess,
        let output,let surface=CVPixelBufferGetIOSurface(output)?.takeUnretainedValue(),
        CVPixelBufferLockBaseAddress(input,.readOnly)==kCVReturnSuccess else {
   throw NSError(domain:"Raw HDMI",code:1)
  }
  defer {CVPixelBufferUnlockBaseAddress(input,.readOnly)}
  let descriptor=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.bgra8Unorm,width:width,height:height,mipmapped:false)
  descriptor.storageMode = .shared
  descriptor.usage = .shaderRead
  guard let texture=device.makeTexture(descriptor:descriptor,iosurface:surface,plane:0),
        let bytes=CVPixelBufferGetBaseAddress(input) else {throw NSError(domain:"Raw HDMI",code:2)}
  texture.replace(region:MTLRegionMake2D(0,0,width,height),mipmapLevel:0,withBytes:bytes,bytesPerRow:CVPixelBufferGetBytesPerRow(input))
  let elapsed=(ProcessInfo.processInfo.systemUptime-start)*1000
  let frame=ProcessedFrame(output,texture,sequence,pts,elapsed,0)
  frame.stages=["rawHDMIBypass":1,"captureWidth":Double(width),"captureHeight":Double(height),"rawCopyMS":elapsed]
  return frame
 }
 private func drain() {
  while true {
   lock.lock()
   guard !stopped,let work=pending else {pending=nil;state.busy=false;lock.unlock();return}
   pending=nil;let config=configuration;lock.unlock()
   autoreleasepool {
    do {
     let began=ProcessInfo.processInfo.systemUptime
     let result:ProcessedFrame
     if config.appliesNR {
     if engineConfiguration?.mode != config.mode || engineConfiguration?.model71 != config.model71 || engineConfiguration?.model31 != config.model31 {
      engine=nil
      engine=try UpstreamNREngine(device:device,configuration:config)
     } else {engine?.configure(config)}
     engineConfiguration=config
     if engineEpoch != work.3 {engine?.resetHistory();engineEpoch=work.3}
     result=try engine!.process(work.0,work.1,work.2,captureCorrection:pairEvidence.wantsCorrection(config,work.3))
     } else {
      engine=nil;engineConfiguration=nil
      _ = pairEvidence.wantsCorrection(config,work.3)
      result=try rawFrame(work.0,sequence:work.1,pts:work.2)
     }
     // Playback retains output surfaces only. Holding capture buffers across
     // the jitter reserve can exhaust AVFoundation's finite capture pool.
     result.sourcePixels=nil;result.motionGuide=work.4
     let ready=ProcessInfo.processInfo.systemUptime
     lock.lock()
     var startMFG=false
     if !stopped && work.3==epoch {
      state.completed &+= 1
      if config.appliesNR {state.neuralCompleted &+= 1;state.appearanceCompleted &+= 1}
      state.error=""
      if let replaced=pendingMFG {
       mfgSuperseded &+= 1
       if let index=history.lastIndex(where:{($0["sequence"] as? UInt64)==replaced.result.sequence}) {
        history[index]["publicationState"]="superseded"
       }
      }
      history.append(["mode":config.mode,"sequence":work.1,"nrStartedAt":began,"completedAt":ready,"processingMS":result.processingMS,
                      "mfgMS":0.0,"workMS":(ready-began)*1000,"generated":0,"pairReset":false,
                      "stages":result.stages,"publicationState":"pendingMFG"])
      if history.count>120 {history.removeFirst(history.count-120)}
      pendingMFG=MFGWork(raw:work.0,result:result,config:config,epoch:work.3,began:began,nrReady:ready)
      startMFG = !mfgBusy
      mfgBusy=true
     }
     lock.unlock()
     if startMFG {mfgWorker.async {[self] in
      // MLX frame generation gets a task-local stream; its mutable graph and
      // model state are owned exclusively by this serial MFG worker.
      Stream.withNewDefaultStream(device:.gpu) {drainMFG()}
     }}
    } catch {
     lock.lock()
     if work.3==epoch {state.failed &+= 1;state.error=String(describing:error)}
     lock.unlock()
    }
   }
  }
 }
 private func drainMFG() {
  while true {
   lock.lock()
   guard !stopped,let job=pendingMFG else {pendingMFG=nil;mfgBusy=false;lock.unlock();return}
   pendingMFG=nil
   let valid=job.epoch==epoch
   lock.unlock()
   guard valid else {continue}
   autoreleasepool {
    let result=job.result,config=job.config
    let work=(job.raw,result.sequence,result.sourceSeconds,job.epoch)
    let mfgStart=ProcessInfo.processInfo.systemUptime
     var generated:[ProcessedFrame]=[],fgMS=0.0,reset=false,difference=0.0,fgError="",parity = -1
     if config.frameGenerationFactor>1 {
      do {
       let output:MultiFrameInterpolator.Result
       if config.frameGenerationBackend=="rife" {
        if rifeInterpolator==nil {
         guard let path=config.rifeModelPath else {throw NSError(domain:"RIFE Metal model",code:1)}
         let model=URL(fileURLWithPath:path)
         rifeInterpolator=try RifeFrameInterpolator(device:device,modelURL:model)
        }
        if rifeEpoch != work.3 {rifeInterpolator?.reset();rifeEpoch=work.3}
        output=try rifeInterpolator!.generate(result,raw:config.appliesNR ? work.0 : nil,factor:config.frameGenerationFactor)
       } else if config.frameGenerationBackend=="apple" {
        if #available(macOS 15.4,*) {
         if appleInterpolator==nil {appleInterpolator=AppleFGService(device:device)}
         if let native=appleInterpolator?.ready() as? AppleFrameInterpolator {
          if appleEpoch != work.3 {native.reset();appleEpoch=work.3}
          output=try native.generate(result,factor:config.frameGenerationFactor)
         } else {output = .init(frames:[],milliseconds:0,cut:true,difference:0,validationMaxDelta:-1)}
        } else {throw NSError(domain:"AppleFG",code:1)}
       } else {
       if interpolator==nil || interpolatorPath != config.frameGenerationPath {
        interpolator=try MultiFrameInterpolator(device:device,path:config.frameGenerationPath)
        interpolatorPath=config.frameGenerationPath
       }
       interpolator?.setResolution(height:config.frameGenerationHeight)
       if interpolatorEpoch != work.3 {interpolator?.reset();interpolatorEpoch=work.3}
       output=try interpolator!.generate(result,factor:config.frameGenerationFactor)
       }
       generated=output.frames;fgMS=output.milliseconds;reset=output.cut;difference=output.difference;parity=output.validationMaxDelta
      } catch {fgError=String(describing:error);rifeInterpolator?.reset();interpolator?.reset();if #available(macOS 15.4,*),let native=appleInterpolator?.ready() as? AppleFrameInterpolator {native.reset()};reset=true}
     } else {interpolator?.reset();interpolatorEpoch=UInt64.max}
     result.nativeFlow=nil
     pairEvidence.submit(work.0,result,config,work.3,generated,[])
     result.correctionEvidence=nil
     let ready=ProcessInfo.processInfo.systemUptime
     lock.lock()
     if !stopped && work.3==epoch {
      let previousSource=frame?.sourceSeconds
      for packet in generated+[result] {nextPresentationID &+= 1;packet.presentationID=nextPresentationID}
      if config.frameGenerationFactor>1 {
       schedule.enqueue(generated+[result],at:ready,interval:previousSource.map{result.sourceSeconds-$0} ?? 0,reset:reset,sourceEnd:result.sourceSeconds,sourceTimes:(generated+[result]).map{$0.sourceSeconds})
       generatedCount &+= UInt64(generated.count);guidedCount &+= UInt64(generated.filter{$0.captureGuided}.count)
       if reset {pairResets &+= 1}
       if !fgError.isEmpty {interpolationFailures &+= 1}
       interpolationError=fgError
      if parity>=0 {warpParity=parity}
      }
      frame=result;published &+= 1
      cachedNativeState=appleInterpolator?.status() ?? ("inactive", "")
      if #available(macOS 15.4,*),let native=appleInterpolator?.ready() as? AppleFrameInterpolator {cachedNativeSkipped=native.skippedBusyPairs}
      if let index=history.lastIndex(where:{($0["sequence"] as? UInt64)==work.1}) {
       history[index]["mfgMS"]=fgMS;history[index]["workMS"]=(ready-job.began)*1000
       history[index]["mfgQueueMS"]=(mfgStart-job.nrReady)*1000
       history[index]["mfgStartedAt"]=mfgStart
       history[index]["publishedAt"]=ready;history[index]["generated"]=generated.count
       history[index]["pairReset"]=reset;history[index]["pairDifference"]=difference
       history[index]["warpParityMaxByteDelta"]=parity;history[index]["stages"]=result.stages
       history[index]["publicationState"]="published"
      }
     }
     lock.unlock()
   }
  }
 }
 public func stop() {
  lock.lock();stopped=true;epoch &+= 1;guideRing.clear(epoch:epoch)
  pending=nil;pendingMFG=nil;frame=nil;schedule.clear();lock.unlock()
  Self.worker.sync {engine=nil;engineConfiguration=nil}
  mfgWorker.sync {interpolator=nil;appleInterpolator=nil;rifeInterpolator=nil}
 }
 package func status()->[String:Any] {
  lock.lock();defer{lock.unlock()}
  let rows=history.filter{($0["mode"] as? String)==configuration.mode}
  let recent=Array(rows.suffix(60)),times=recent.compactMap{$0["processingMS"] as? Double}.sorted()
  let activeSettings=(try? JSONSerialization.jsonObject(with:JSONEncoder().encode(configuration))) ?? [:]
  let first=recent.first?["completedAt"] as? Double ?? 0,last=recent.last?["completedAt"] as? Double ?? 0
  let nativeSkipped=cachedNativeSkipped
  let nativeState=cachedNativeState
  var display=nativePresentation
  if let sampled=display["sampledAtHostSeconds"] as? Double,ProcessInfo.processInfo.systemUptime-sampled>2.5 {display["active"]=false;display["baseShownFPS"]=0;display["generatedShownFPS"]=0;display["uniqueShownFPS"]=0}
  return ["execution":"bounded-overlapped-nr-mfg","nrWorkerBusy":state.busy,"mfgWorkerBusy":mfgBusy,"mfgPending":pendingMFG != nil,"mfgSuperseded":mfgSuperseded,"publishedBaseFrames":published,"captureGuides":guideRing.status(),"captureGuided":false,"nrImplementation":"obs-dlss5-nr-macos","legacyDisplayControlsActive":false,"guidedFrames":guidedCount,"nativeFGState":nativeState.0,"nativeFGError":nativeState.1,"nativeFGSkippedPairs":nativeSkipped,"mfgBackend":configuration.frameGenerationBackend,"mfgCompositor":configuration.frameGenerationBackend=="rife" ? "rife-v4.25-fast" : configuration.frameGenerationBackend=="apple" ? "apple-native-flow-single-anchor" : ProcessInfo.processInfo.environment["DLSS_MFG_LEGACY"]=="1" ? "legacy-phase-blend" : "single-trajectory-single-source","nativePreview":display,"settings":activeSettings,"mode":configuration.title,"modeID":configuration.mode,"backend":"obs-dlss5-nr native MLX FP16; MFG \(configuration.frameGenerationFactor)x \(configuration.frameGenerationBackend=="apple" ? "Apple VT, system-managed hardware" : configuration.frameGenerationBackend=="rife" ? "RIFE Metal v4.25" : "MLX/Metal")","submitted":state.submitted,"completed":state.completed,"replaced":state.superseded,"failed":state.failed,"error":state.error,"busy":state.busy,"processingMedianMS":times.isEmpty ? 0 : times[times.count/2],"processedFPS":last>first ? Double(recent.count-1)/(last-first) : 0,"warpParityMaxByteDelta":warpParity,"mfgFlowHeight":configuration.frameGenerationHeight,"mfgFactor":configuration.frameGenerationFactor,"generatedFrames":generatedCount,"mfgFailures":interpolationFailures,"mfgError":interpolationError,"pairResets":pairResets,"playbackQueueDepth":schedule.slots.count,"playbackMaxQueueDepth":schedule.maxDepth,"playbackDebtResets":schedule.debtResets,"playbackUnderruns":schedule.underruns,"playbackClockDelayMS":schedule.clockDelayMS,"expiredBeforeSelection":schedule.expiredBeforeSelection,"selectedFrameAgeMS":schedule.selected.map{(ProcessInfo.processInfo.systemUptime-$0.sourceSeconds)*1000} ?? 0,"outputWidth":frame?.texture.width ?? 0,"outputHeight":frame?.texture.height ?? 0,"stages":frame?.stages ?? [:],"recentFrames":history]
 }
}
