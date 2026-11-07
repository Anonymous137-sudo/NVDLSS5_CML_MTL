import Foundation
import CoreVideo
import Metal
import MetalPerformanceShaders
import DLSSCore
import DLSSMLX
import MLX

// MLX0.31 uses the colocated library in this isolated app; never load the
// existing MLX0.32 plugin runtime in this process.
func npSetMetallib(_ ignored:UnsafePointer<CChar>) {}
public final class ProcessedFrame {
 public let pixels:CVPixelBuffer,texture:MTLTexture,sequence:UInt64,sourceSeconds:Double
 public let processingMS:Double,gpuMS:Double,aneMS:Double
 public var validation=false
 public var sourcePixels:CVPixelBuffer?
 public var stages:[String:Double]=[:]
 public var variants:[String:CVPixelBuffer]=[:]
 public let neural=true,appearance=true
 init(_ p:CVPixelBuffer,_ t:MTLTexture,_ seq:UInt64,_ pts:Double,_ ms:Double,_ gpu:Double,_ ane:Double=0) { pixels=p;texture=t;sequence=seq;sourceSeconds=pts;processingMS=ms;gpuMS=gpu;aneMS=ane }
}
public struct LiveStats {
 public var submitted:UInt64=0,superseded:UInt64=0,completed:UInt64=0,neuralCompleted:UInt64=0,appearanceCompleted:UInt64=0,failed:UInt64=0
 public var busy=false
 public var error=""
}
public final class PostPipeline {
 public let device:MTLDevice
 public let experimentalAppearance=false
 public var onComplete:((ProcessedFrame)->Void)?
 private let lock=NSLock()
 private var state=LiveStats(),pending:(CVPixelBuffer,UInt64,Double,UInt64)?,latestFrame:ProcessedFrame?
 private var epoch:UInt64=0,stopped=false
 private let renderer:MLXNeuralRenderer?,sr:CompactSR
 private var experiment:ThreeModeEngine?
 private let commands:MTLCommandQueue,scale:MPSImageLanczosScale
 private let lowTexture:MTLTexture,highStorage:MTLBuffer,lowStorage:MTLBuffer
 private let unpack:MTLComputePipelineState
 private let lw:Int,lh:Int
 private let split:MLXStageSplit?,island:LiveANEIsland?
 private var processed:UInt64=0
 private var validatedANE=false,islandRejected=false
 private var firstGPUValidation:[String:Any]=[:],firstANEValidation:[String:Any]=[:]
 private var stageRecords:[[String:Any]]=[]
 public var modeLabel:String {
  if let experiment {return (experiment.mode.hasSuffix("-480") ? "" : "720p · ") + experiment.label}
  if split == nil {return "144p full model"}
  if islandRejected {return "720p GPU control · ANE quality gate rejected"}
  return island?.ready == true ? "720p full model · CPU+ANE bottleneck" : "720p full model · GPU control"
 }
 public var diagnostics:[String:Any] {
  lock.lock();let records=stageRecords,gpuCheck=firstGPUValidation,aneCheck=firstANEValidation,rejected=islandRejected;lock.unlock()
  return ["mode":experiment?.mode ?? (split == nil ? "full-model144p" : "full-model720p-stage-split"), "island":island?.report ?? ["phase":"GPU control"],"recentFrames":records,"firstGPUValidation":gpuCheck,"firstANEValidation":aneCheck,"aneQualityRejected":rejected]
 }
 private let geometry:NeuralRenderingNetworkGeometry
 private let guide:MLXFast.MLXFastKernel,pack:MLXFast.MLXFastKernel,featureKernel:MLXFast.MLXFastKernel
 private var featureTemplate:MLXArray?,featureHost:HostTensor?
 public init(modelURL:URL?,aneURL:URL?=nil) throws {
  guard aneURL==nil,let modelURL,let gpu=MTLCreateSystemDefaultDevice(),let commands=gpu.makeCommandQueue(),
   let path=CommandLine.arguments.first(where:{$0.hasPrefix("--full-model=")}) else { throw NSError(domain:"LiveQuality",code:1) }
  device=gpu;self.commands=commands;scale=MPSImageLanczosScale(device:gpu)
  let splitPath=CommandLine.arguments.first(where:{$0.hasPrefix("--split-model=")}).map{String($0.dropFirst(14))}
  lw=splitPath == nil ? 256 : 1280;lh=splitPath == nil ? 144 : 720
  split=try splitPath.map{_ in try MLXStageSplit(packageURL:URL(fileURLWithPath:String(path.dropFirst(13))))}
  island=CommandLine.arguments.contains("--split-gpu-only") ? nil : splitPath.map{LiveANEIsland(url:URL(fileURLWithPath:$0))}
  geometry=try NeuralRenderingNetworkGeometry(outputWidth:lw,outputHeight:lh,networkWidth:splitPath == nil ? 256 : 1280,networkHeight:splitPath == nil ? 192 : 768)
  renderer=CommandLine.arguments.contains(where:{$0.hasPrefix("--experiment-mode=")}) ? nil : try MLXNeuralRenderer(packageURL:URL(fileURLWithPath:String(path.dropFirst(13))),executionMode:.metalFused,computePrecision:.float16)
  sr=try CompactSR(url:modelURL)
  let td=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba32Float,width:lw,height:lh,mipmapped:false);td.storageMode = .shared;td.usage=[.shaderRead,.shaderWrite]
  guard let texture=gpu.makeTexture(descriptor:td),let high=gpu.makeBuffer(length:1280*720*3*2,options:.storageModeShared),let low=gpu.makeBuffer(length:lw*lh*3*4,options:.storageModeShared) else { throw NSError(domain:"LiveQuality",code:2) }
  lowTexture=texture;highStorage=high;lowStorage=low
  let lib=try gpu.makeLibrary(source:"""
  #include <metal_stdlib>
  using namespace metal;
  kernel void unpack(texture2d<float,access::read> src [[texture(0)]],texture2d<float,access::read> low [[texture(1)]],device half* high [[buffer(0)]],device float* outLow [[buffer(1)]],uint2 p [[thread_position_in_grid]]) {
   if(p.x<1280 && p.y<720) { float3 v=src.read(p).rgb;uint i=(p.y*1280+p.x)*3;for(uint c=0;c<3;c++) high[i+c]=half(v[c]); }
   if(p.x<low.get_width() && p.y<low.get_height()) { float3 v=low.read(p).rgb;uint i=(p.y*low.get_width()+p.x)*3;for(uint c=0;c<3;c++) outLow[i+c]=v[c]; }
  }
  """,options:nil)
  unpack=try gpu.makeComputePipelineState(function:lib.makeFunction(name:"unpack")!)
  featureKernel=MLXFast.metalKernel(name:"live_full720_features",inputNames:["base","high"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;if(p>=1280*768) return;
  uint x=p%1280,y=p/1280,sy=y<720 ? y : 1438-y;
  for(uint c=0;c<16;c++) out[p*16+c]=base[p*16+c];
  for(uint c=0;c<3;c++) {
   half sampled=high[(sy*1280+x)*3+c];
   half centered=half(sampled-half(.5f));half value=half(centered*half(.125f));
   out[p*16+4+c]=value;out[p*16+7+c]=value;
  }
  """)
  guide=MLXFast.metalKernel(name:"quality_low_delta",inputNames:["original","styled","high"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;if(p>=1280*720) return;
  uint x=p%1280,y=p/1280;
  float2 q=clamp((float2(x,y)+.5f)*float2(LW/1280.f,LH/720.f)-.5f,float2(0),float2(LW-1,LH-1));
  uint2 a=uint2(q),b=min(a+1,uint2(LW-1,LH-1));float2 f=fract(q);
  bool hud=x<333 && y>=504;
  for(uint c=0;c<3;c++) {
   uint i0=(a.y*LW+a.x)*3+c,i1=(a.y*LW+b.x)*3+c,i2=(b.y*LW+a.x)*3+c,i3=(b.y*LW+b.x)*3+c;
   float d=mix(mix(float(styled[i0])-float(original[i0]),float(styled[i1])-float(original[i1]),f.x),mix(float(styled[i2])-float(original[i2]),float(styled[i3])-float(original[i3]),f.x),f.y);
   out[p*3+c]=clamp(float(high[p*3+c])+(hud ? 0.f : d),0.f,1.f);
  }
  """)
  pack=MLXFast.metalKernel(name:"quality_sr_pack",inputNames:["rgb","bias"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;if(p>=3840*2160) return;
  uint x=p%3840,y=p/3840,sub=(y%6)*6+x%6;
  uint i=((y/6)*640+x/6)*108+sub;
  for(uint c=0;c<3;c++) out[p*4+2-c]=uchar(clamp(round(float(rgb[i+c*36]+bias[sub+c*36])*255.f),0.f,255.f));
  out[p*4+3]=255;
  """)
  if CommandLine.arguments.contains(where:{$0.hasPrefix("--experiment-mode=")}) {experiment=try ThreeModeEngine(device:gpu,srURL:modelURL)}
 }
 public var stats:LiveStats { lock.lock();defer{lock.unlock()};return state }
 public func latest()->ProcessedFrame? { lock.lock();defer{lock.unlock()};return latestFrame }
 public func submit(_ pixels:CVPixelBuffer,sequence:UInt64,sourceSeconds:Double) {
  lock.lock();guard !stopped else {lock.unlock();return}
  state.submitted+=1;if pending != nil {state.superseded+=1};pending=(pixels,sequence,sourceSeconds,epoch)
  let start = !state.busy;if start {state.busy=true};lock.unlock()
  if start { Task.detached(priority:.userInitiated) { await self.drain() } }
 }
 public func reset() { lock.lock();epoch &+= 1;pending=nil;latestFrame=nil;lock.unlock() }
 public func stop() { lock.lock();stopped=true;epoch &+= 1;pending=nil;latestFrame=nil;lock.unlock() }
 private func next()->(CVPixelBuffer,UInt64,Double,UInt64)? {
  lock.lock();defer{lock.unlock()};guard let job=pending else {state.busy=false;return nil};pending=nil;return job
 }
 private func publish(_ frame:ProcessedFrame,_ generation:UInt64) {
  lock.lock();let valid = !stopped && generation==epoch
  if valid {
   latestFrame=frame;state.completed+=1;state.neuralCompleted+=1;state.appearanceCompleted+=1
   var record:[String:Any]=["sequence":frame.sequence,"sourcePTS":frame.sourceSeconds,"completedAt":ProcessInfo.processInfo.systemUptime,"processingMS":frame.processingMS,"aneMS":frame.aneMS,"validation":frame.validation,"stages":frame.stages]
   record["backend"]=experiment != nil ? "MLX Metal-fused " + experiment!.mode : (frame.aneMS>0 ? "cpuAndNeuralEngine" : "GPU control")
   if frame.validation {
    if frame.aneMS>0 && firstANEValidation.isEmpty {firstANEValidation=record}
    if frame.aneMS==0 && firstGPUValidation.isEmpty {firstGPUValidation=record}
   }
   stageRecords.append(record);if stageRecords.count>360 {stageRecords.removeFirst()}
  };lock.unlock()
  if valid {onComplete?(frame)}
 }
 private func failure(_ error:Error) {lock.lock();state.failed+=1;state.error=String(describing:error);lock.unlock()}
 private func drain() async {
  while let job=next() {
   do { let frame=try await process(job.0,job.1,job.2);publish(frame,job.3) } catch {failure(error)}
  }
 }
 private func texture(_ pixels:CVPixelBuffer)throws->MTLTexture {
  guard CVPixelBufferGetPixelFormatType(pixels)==kCVPixelFormatType_32BGRA,let surface=CVPixelBufferGetIOSurface(pixels)?.takeUnretainedValue() else {throw NSError(domain:"LiveQuality",code:3)}
  let d=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.bgra8Unorm,width:CVPixelBufferGetWidth(pixels),height:CVPixelBufferGetHeight(pixels),mipmapped:false)
  d.storageMode = .shared;d.usage=[.shaderRead,.shaderWrite]
  guard let t=device.makeTexture(descriptor:d,iosurface:surface,plane:0) else {throw NSError(domain:"LiveQuality",code:4)};return t
 }
 private func finish(_ cb:MTLCommandBuffer)throws->Double {
  cb.commit();cb.waitUntilCompleted();guard cb.status == .completed else {throw cb.error ?? NSError(domain:"LiveQuality",code:5)}
  return (cb.gpuEndTime-cb.gpuStartTime)*1000
 }
 private func prepare(_ input:CVPixelBuffer)throws->(HostTensor,MLXArray,MLXArray,Double) {
  guard CVPixelBufferGetWidth(input)==1280,CVPixelBufferGetHeight(input)==720 else {throw NSError(domain:"LiveQuality",code:6)}
  let source=try texture(input)
  guard let cb=commands.makeCommandBuffer() else {throw NSError(domain:"LiveQuality",code:7)}
  if split == nil { scale.encode(commandBuffer:cb,sourceTexture:source,destinationTexture:lowTexture) }
  guard let encoder=cb.makeComputeCommandEncoder() else {throw NSError(domain:"LiveQuality",code:8)}
  encoder.setComputePipelineState(unpack);encoder.setTexture(source,index:0);encoder.setTexture(split == nil ? lowTexture : source,index:1);encoder.setBuffer(highStorage,offset:0,index:0);encoder.setBuffer(lowStorage,offset:0,index:1)
  encoder.dispatchThreads(MTLSize(width:1280,height:720,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1));encoder.endEncoding()
  let gpu=try finish(cb)
  let bytes=Data(bytesNoCopy:lowStorage.contents(),count:lw*lh*3*4,deallocator:.none)
  let low=try HostTensor(descriptor:TensorDescriptor(name:"color",shape:[1,lh,lw,3],dataType:.float32,layout:.nhwc),bytes:bytes)
  let lowOwner=lowStorage
  let lowArray = split == nil ? MLXArray(bytes,[1,lh,lw,3],dtype:.float32).asType(.float16) : MLXArray(rawPointer:lowOwner.contents(),[1,lh,lw,3],dtype:.float32){_ = lowOwner}
  let storage=highStorage
  let high=MLXArray(rawPointer:storage.contents(),[1,720,1280,3],dtype:.float16){_ = storage}
  return (low,lowArray,high,gpu)
 }
 private func process(_ input:CVPixelBuffer,_ sequence:UInt64,_ pts:Double) async throws->ProcessedFrame {
  if let experiment {return try experiment.process(input,sequence,pts)}
  let began=ProcessInfo.processInfo.systemUptime
  let (low,originalLow,high,prepGPU)=try autoreleasepool {try prepare(input)}
  processed+=1
  let validateLive=processed==1 || (island?.ready == true && !islandRejected && (!validatedANE || processed%30==0))
  let prepStart=ProcessInfo.processInfo.systemUptime
  let needCPU=split == nil || featureTemplate == nil || validateLive
  let features:HostTensor
  if needCPU {
   features=try autoreleasepool {try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(from:low,noiseFrameIndex:0,geometry:geometry,normalizedStyle:0,localToneStrength:1,localStructureStrength:1,automaticMask:nil,controlMask:nil)}
  } else {features=featureHost!}
  var aneMS=0.0,validation=false,stages:[String:Double]=["featurePreparationMS":(ProcessInfo.processInfo.systemUptime-prepStart)*1000]
  var raw:HostTensor?
  var fullAppearance:MLXArray?
  if let split {
   let gpuFeatureStart=ProcessInfo.processInfo.systemUptime
   if featureTemplate == nil {
    featureHost=features
    featureTemplate=MLXArray(features.bytes,[1,768,1280,16],dtype:.float32).asType(.float16);eval(featureTemplate!)
   }
   let tensor=featureKernel([featureTemplate!,high],grid:(1280*768,1,1),threadGroup:(256,1,1),outputShapes:[[1,768,1280,16]],outputDTypes:[.float16])[0]
   if needCPU {
    let oracle=MLXArray(features.bytes,[1,768,1280,16],dtype:.float32).asType(.float16)
    let error=Double(max(abs(tensor-oracle)).item(Float.self))
    stages["liveFeatureMaxAbs"]=error
    guard error==0 else {throw NSError(domain:"LiveQuality",code:13,userInfo:[NSLocalizedDescriptionKey:"Live feature parity failed: \(error)"])}
   }
   eval(tensor)
   stages["gpuFeatureAssemblyWallMS"]=(ProcessInfo.processInfo.systemUptime-gpuFeatureStart)*1000
   let encoderStart=ProcessInfo.processInfo.systemUptime
   let context=split.begin(tensor)
   stages["gpuEncoderWallMS"]=(ProcessInfo.processInfo.systemUptime-encoderStart)*1000
   let center:MLXArray
   if let island,island.ready,!islandRejected {
    if CommandLine.arguments.contains("--split-global") {
     let middle=split.beginGlobal(context)
     let result=try await island.predict(middle.input)
     center=split.finishGlobal(middle,latent:result.0);eval(center);aneMS=result.1
    } else {let result=try await island.predict(context.centerInput);center=result.0;aneMS=result.1}
   } else { center=split.referenceCenter(context);eval(center) }
   stages["centerTransferAndInferenceMS"]=aneMS
   let decoderStart=ProcessInfo.processInfo.systemUptime
   var output=split.finish(context,center:center);eval(output)
   stages["gpuDecoderWallMS"]=(ProcessInfo.processInfo.systemUptime-decoderStart)*1000
   // Paired comparisons use this exact captured frame, never a cached PNG.
   validation=validateLive
   if validation {
    let referenceStart=ProcessInfo.processInfo.systemUptime
    let result=try await renderer!.render(NeuralRenderRequest(sequenceID:sequence,inputs:[features]))
    guard let baseline=result.output(named:"color") else {throw NSError(domain:"LiveQuality",code:12)}
    let expected=MLXArray(baseline.bytes,[1,768,1280,4],dtype:.float32)
    let diff=abs(output.asType(.float32)-expected)
    let actualRGB=clip(high.asType(.float32)+output[0...,0..<720,0...,0..<3].asType(.float32)*0.25,min:0,max:1)
    let expectedRGB=clip(high.asType(.float32)+expected[0...,0..<720,0...,0..<3]*0.25,min:0,max:1)
    let rgbDiff=abs(actualRGB-expectedRGB)
    stages["composedRGBMAE"]=Double(mean(rgbDiff).item(Float.self))
    stages["composedRGBMaxAbs"]=Double(max(rgbDiff).item(Float.self))
    if aneMS>0 {
     if stages["composedRGBMAE"]!>0.005 || stages["composedRGBMaxAbs"]!>0.1 {
      lock.lock();islandRejected=true;lock.unlock()
      stages["rejectedANETransferMS"]=aneMS;aneMS=0
      output=expected.asType(.float16)
     } else {validatedANE=true}
    }
    stages["rawHeadMAE"]=Double(mean(diff).item(Float.self))
    stages["rawHeadMaxAbs"]=Double(max(diff).item(Float.self))
    stages["validationWallMS"]=(ProcessInfo.processInfo.systemUptime-referenceStart)*1000
   }
   guard all(isFinite(output)).item(Bool.self) else {throw NSError(domain:"LiveQuality",code:14,userInfo:[NSLocalizedDescriptionKey:"Nonfinite GPU head"])}
   fullAppearance=clip(originalLow+output[0...,0..<720,0...,0..<3].asType(.float32)*0.25,min:0,max:1).asType(.float16)
  } else {
   let result=try await renderer!.render(NeuralRenderRequest(sequenceID:sequence,inputs:[features]))
   guard let value=result.output(named:"color") else {throw NSError(domain:"LiveQuality",code:9)}
   raw=value
  }
  return try autoreleasepool {
   let appearance:MLXArray
   if let fullAppearance { appearance=fullAppearance }
   else {
    guard let raw else {throw NSError(domain:"LiveQuality",code:15)}
    let head=try geometry.cropOutput(raw)
    let styled=try NeuralRenderingFirstFramePostprocessor.compose(head:head,over:low,controlMask:nil,intensity:1)
    guard styled.bytes.withUnsafeBytes({$0.bindMemory(to:Float.self).allSatisfy{$0.isFinite}}) else {throw NSError(domain:"LiveQuality",code:10)}
    appearance=MLXArray(styled.bytes,[1,lh,lw,3],dtype:.float32).asType(.float16)
   }
   let image = split != nil ? appearance : guide([originalLow,appearance,high],template:[("LW",256),("LH",144)],grid:(1280*720,1,1),threadGroup:(256,1,1),outputShapes:[[1,720,1280,3]],outputDTypes:[.float16])[0]
   let rgb=sr.head(image)
   let packed=pack([rgb,sr.outputBias],grid:(3840*2160,1,1),threadGroup:(256,1,1),outputShapes:[[2160,3840,4]],outputDTypes:[.uint8])[0]
   eval(packed)
   var output:CVPixelBuffer?
   let attrs:[CFString:Any]=[kCVPixelBufferIOSurfacePropertiesKey:[:],kCVPixelBufferMetalCompatibilityKey:true]
   guard CVPixelBufferCreate(nil,3840,2160,kCVPixelFormatType_32BGRA,attrs as CFDictionary,&output)==kCVReturnSuccess,let output,let buffer=packed.asMTLBuffer(device:device,noCopy:true),let cb=commands.makeCommandBuffer(),let blit=cb.makeBlitCommandEncoder() else {throw NSError(domain:"LiveQuality",code:11)}
   let target=try texture(output)
   blit.copy(from:buffer,sourceOffset:0,sourceBytesPerRow:3840*4,sourceBytesPerImage:3840*2160*4,sourceSize:MTLSize(width:3840,height:2160,depth:1),to:target,destinationSlice:0,destinationLevel:0,destinationOrigin:MTLOrigin(x:0,y:0,z:0));blit.endEncoding()
   let gpu=try finish(cb);withExtendedLifetime((packed,buffer)) {}
   let frame=ProcessedFrame(output,target,sequence,pts,(ProcessInfo.processInfo.systemUptime-began)*1000,prepGPU+gpu,aneMS)
   frame.validation=validation;frame.sourcePixels=input;frame.stages=stages
   return frame
  }
 }
}
