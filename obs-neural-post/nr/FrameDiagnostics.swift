import Foundation
import CoreVideo
import CoreGraphics
import ImageIO
import Metal

/// Opt-in, bounded live-pair evidence. PNG encoding never runs on inference,
/// capture or OBS render threads. The capture pool buffer is never retained.
final class LiveFrameDiagnostics {
 private let captureStateLock=NSLock()
 private let queue=DispatchQueue(label:"obs.nr.pair-evidence",qos:.utility),lock=NSLock()
 private var busy=false,lastEpoch:UInt64=UInt64.max,count=0,lastTime=0.0,total=0
 private var previous:ProcessedFrame?
 private let directory=ProcessInfo.processInfo.environment["DLSS_NR_PAIR_DUMP"]
 private struct Image {let data:Data,width:Int,height:Int,stride:Int}
 private let burstRoot=ProcessInfo.processInfo.environment["DLSS_NR_BURST_DIR"].map { URL(fileURLWithPath:$0) }
 private struct BurstFrame {
  let raw: Image
  let output: Image
  let sequence: UInt64
  let pts: Double
  let stages: [String:Double]
  let correction: CorrectionEvidence?
 }
 private var burst: [BurstFrame]=[]
 private var burstEpoch: UInt64?
 private var burstID: UUID?

 // Called on the inference worker before processing so GPU readback is
 // scheduled only for frames requested by an active burst.
 func wantsCorrection(_ config:NRConfiguration,_ epoch:UInt64)->Bool {
  captureStateLock.lock();defer{captureStateLock.unlock()}
  guard let root=burstRoot else{return false}
  if burstEpoch != epoch || config.frameGenerationFactor > 1 {
   burst.removeAll()
   burstID=nil
   burstEpoch=epoch
  }
  guard config.frameGenerationFactor<=1 else{return false}
  let trigger=root.appendingPathComponent("capture-base")
  if burstID == nil, (try? FileManager.default.removeItem(at:trigger)) != nil {burstID=UUID()}
  return burstID != nil
 }
 private func captureBurst(_ input:CVPixelBuffer,_ output:ProcessedFrame,_ config:NRConfiguration) {
  guard let root=burstRoot else{return}
  guard let id=burstID,let raw=copy(input),let processed=copy(output.pixels) else{return}
  burst.append(BurstFrame(raw:raw,output:processed,sequence:output.sequence,pts:output.sourceSeconds,stages:output.stages,correction:output.correctionEvidence))
  guard burst.count==12 else{return}
  // Keep at most one completed batch in flight. No capture buffers are retained.
  lock.lock()
  guard !busy else {lock.unlock();burst.removeAll();burstID=nil;return}
  busy=true
  lock.unlock()
  let batch=burst
  burst.removeAll()
  burstID=nil
  queue.async { [self] in
   defer {lock.lock();busy=false;lock.unlock()}
   do {
    let directory=root.appendingPathComponent(id.uuidString)
    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
    let settings=try JSONSerialization.jsonObject(with:JSONEncoder().encode(config))
    var frames:[[String:Any]]=[]
    for (index,item) in batch.enumerated() {
     let prefix=String(format:"%02d",index)
     try write(item.raw,to:directory.appendingPathComponent(prefix+"-raw.png"))
     try write(item.output,to:directory.appendingPathComponent(prefix+"-nr.png"))
     var frame:[String:Any]=["sequence":item.sequence,"sourceSeconds":item.pts,"stages":item.stages]
     if let c=item.correction {
      try c.model.write(to:directory.appendingPathComponent(prefix+"-model.rgba16f"))
      try c.before.write(to:directory.appendingPathComponent(prefix+"-before.rgba16f"))
      try c.after.write(to:directory.appendingPathComponent(prefix+"-after.rgba16f"))
      frame["correction"]=["width":c.width,"height":c.height,"bytesPerRow":c.rowBytes]
     }
     frames.append(frame)
    }
    try JSONSerialization.data(withJSONObject:["settings":settings,"frames":frames],options:[.prettyPrinted,.sortedKeys])
     .write(to:directory.appendingPathComponent("sequence.json"),options:.atomic)
   } catch {NSLog("NR burst evidence: %@",String(describing:error))}
  }
 }
 private func write(_ image:Image,to url:URL)throws {
  guard let provider=CGDataProvider(data:image.data as CFData),
        let color=CGColorSpace(name:CGColorSpace.sRGB),
        let cg=CGImage(width:image.width,height:image.height,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:image.stride,
                       space:color,bitmapInfo:CGBitmapInfo(rawValue:CGBitmapInfo.byteOrder32Little.rawValue|CGImageAlphaInfo.noneSkipFirst.rawValue),
                       provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent),
        let destination=CGImageDestinationCreateWithURL(url as CFURL,"public.png" as CFString,1,nil) else {
   throw CocoaError(.fileWriteUnknown)
  }
  CGImageDestinationAddImage(destination,cg,nil)
  guard CGImageDestinationFinalize(destination) else {throw CocoaError(.fileWriteUnknown)}
 }
 private func copy(_ p:CVPixelBuffer)->Image? {
  guard CVPixelBufferGetPixelFormatType(p)==kCVPixelFormatType_32BGRA,CVPixelBufferLockBaseAddress(p,.readOnly)==kCVReturnSuccess else{return nil}
  defer{CVPixelBufferUnlockBaseAddress(p,.readOnly)}
  guard let base=CVPixelBufferGetBaseAddress(p) else{return nil}
  let w=CVPixelBufferGetWidth(p),h=CVPixelBufferGetHeight(p),row=CVPixelBufferGetBytesPerRow(p)
  return Image(data:Data(bytes:base,count:row*h),width:w,height:h,stride:row)
 }
 private func copy(_ t:MTLTexture)->Image? {
  guard t.pixelFormat == .bgra8Unorm else{return nil}
  let row=t.width*4;var data=Data(count:row*t.height)
  data.withUnsafeMutableBytes{t.getBytes($0.baseAddress!,bytesPerRow:row,from:MTLRegionMake2D(0,0,t.width,t.height),mipmapLevel:0)}
  return Image(data:data,width:t.width,height:t.height,stride:row)
 }
 func submit(_ input:CVPixelBuffer,_ output:ProcessedFrame,_ config:NRConfiguration,_ epoch:UInt64,_ generated:[ProcessedFrame],_ guides:[CaptureGuide]) {
  captureStateLock.lock();defer{captureStateLock.unlock()}
  captureBurst(input,output,config)
  guard let directory else{return}
  let now=ProcessInfo.processInfo.systemUptime
  lock.lock();if lastEpoch != epoch {lastEpoch=epoch;count=0;lastTime=0;previous=nil}
  let prior=previous;previous=output
  guard (config.frameGenerationFactor<=1 || !generated.isEmpty),!busy,count<3,total<24,now-lastTime>1 else{lock.unlock();return};busy=true;count+=1;total+=1;lastTime=now;lock.unlock()
  guard let raw=copy(input),let nr=copy(output.pixels) else{lock.lock();busy=false;lock.unlock();return}
  var images:[(String,Image)]=[("capture",raw),("base-output",nr)]
  if let prior,let image=copy(prior.pixels) {images.append(("previous-base",image));if let guide=prior.motionGuide,let raw=copy(guide.texture){images.append(("previous-capture",raw))}}
  let intermediate=generated.isEmpty ? nil:generated[generated.count/2]
  if let intermediate,let image=copy(intermediate.pixels) {
   images.append(("mfg-intermediate",image))
   if let guide=guides.min(by:{abs($0.seconds-intermediate.sourceSeconds)<abs($1.seconds-intermediate.sourceSeconds)}),abs(guide.seconds-intermediate.sourceSeconds)<0.0001,let image=copy(guide.texture){images.append(("actual-intermediate-capture",image))}
  }
  let sequence=output.sequence,source=output.sourceSeconds,stages=output.stages
  let intermediateTime=intermediate?.sourceSeconds,intermediatePhase=intermediate?.phase
  queue.async {[self] in
   defer{lock.lock();busy=false;lock.unlock()}
   do {
    let dir=URL(fileURLWithPath:directory).appendingPathComponent("epoch-\(epoch)-seq-\(sequence)")
    try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true)
    for (name,image) in images {
     try write(image,to:dir.appendingPathComponent(name+".png"))
    }
    let settings=try JSONSerialization.jsonObject(with:JSONEncoder().encode(config))
    var metadata:[String:Any]=["sequence":sequence,"sourceSeconds":source,"captureWidth":raw.width,"captureHeight":raw.height,"settings":settings,"stages":stages,"input":"exact capture for this base output; no interpolated frame"]
    if let intermediateTime,let intermediatePhase {metadata["intermediateSourceSeconds"]=intermediateTime;metadata["intermediatePhase"]=intermediatePhase}
    try JSONSerialization.data(withJSONObject:metadata,options:[.prettyPrinted,.sortedKeys]).write(to:dir.appendingPathComponent("pair.json"))
   } catch {NSLog("NR pair evidence: %@",String(describing:error))}
  }
 }
}
