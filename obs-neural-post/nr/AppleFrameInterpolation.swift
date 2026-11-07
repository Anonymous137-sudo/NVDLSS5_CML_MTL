import Foundation
import CoreVideo
import CoreMedia
import Metal
import VideoToolbox

/// Native raw optical flow, then a single current NR appearance anchor on each
/// real capture guide. No independent styled/raw FRC color reconstructions.
@available(macOS 15.4, *)
final class AppleFrameInterpolator {
 private let processor=VTFrameProcessor(),device:MTLDevice,queue:MTLCommandQueue
 private let toHalf:MTLComputePipelineState,pack:MTLComputePipelineState
 private let flowAttributes:[String:Any],fw:Int,fh:Int
 private struct Endpoint {let frame:ProcessedFrame,raw:(CVPixelBuffer,MTLTexture)}
 private var previous:Endpoint?,random=true
 private final class Flight {
  let lock=NSLock(),done=DispatchSemaphore(value:0);var finished=false,error:Error?
  func complete(_ error:Error?){lock.lock();self.error=error;finished=true;lock.unlock();done.signal()}
  func snapshot()->(Bool,Error?){lock.lock();defer{lock.unlock()};return(finished,error)}
 }
 private var flight:Flight?
 private let stateLock=NSLock();private var skipped:UInt64=0
 var skippedBusyPairs:UInt64 {stateLock.lock();defer{stateLock.unlock()};return skipped}
 private func skip(){stateLock.lock();skipped &+= 1;stateLock.unlock()}
 init(device:MTLDevice)throws {
  guard VTOpticalFlowConfiguration.isSupported,let config=VTOpticalFlowConfiguration(frameWidth:1920,frameHeight:1080,qualityPrioritization:.normal,revision:.revision1),let queue=device.makeCommandQueue(),let fw=config.destinationPixelBufferAttributes[kCVPixelBufferWidthKey as String] as? Int,let fh=config.destinationPixelBufferAttributes[kCVPixelBufferHeightKey as String] as? Int else{throw NSError(domain:"AppleFG",code:1)}
  self.device=device;self.queue=queue;self.fw=fw;self.fh=fh;flowAttributes=config.destinationPixelBufferAttributes
  let lib=try device.makeLibrary(source:Self.shader,options:nil)
  toHalf=try device.makeComputePipelineState(function:lib.makeFunction(name:"to_half")!);pack=try device.makeComputePipelineState(function:lib.makeFunction(name:"anchor_pack")!)
  try processor.startSession(configuration:config)
 }
 deinit{processor.endSession()}
 func reset(){previous=nil;random=true}
 private func surface(half:Bool)throws->(CVPixelBuffer,MTLTexture) {
  var p:CVPixelBuffer?;let attrs:[CFString:Any]=[kCVPixelBufferIOSurfacePropertiesKey:[:],kCVPixelBufferMetalCompatibilityKey:true]
  guard CVPixelBufferCreate(nil,1920,1080,half ? kCVPixelFormatType_64RGBAHalf:kCVPixelFormatType_32BGRA,attrs as CFDictionary,&p)==kCVReturnSuccess,let p,let io=CVPixelBufferGetIOSurface(p)?.takeUnretainedValue() else{throw NSError(domain:"AppleFG",code:2)}
  CVBufferSetAttachment(p,kCVImageBufferColorPrimariesKey,kCVImageBufferColorPrimaries_ITU_R_709_2,.shouldPropagate)
  CVBufferSetAttachment(p,kCVImageBufferTransferFunctionKey,kCVImageBufferTransferFunction_sRGB,.shouldPropagate)
  let td=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:half ? .rgba16Float:.bgra8Unorm,width:1920,height:1080,mipmapped:false);td.storageMode = .shared;td.usage=[.shaderRead,.shaderWrite]
  guard let t=device.makeTexture(descriptor:td,iosurface:io,plane:0) else{throw NSError(domain:"AppleFG",code:3)};return(p,t)
 }
 private func flowBuffer()throws->(CVPixelBuffer,MTLTexture) {
  var p:CVPixelBuffer?;var attrs=flowAttributes;attrs[kCVPixelBufferIOSurfacePropertiesKey as String]=[:];attrs[kCVPixelBufferMetalCompatibilityKey as String]=true
  guard CVPixelBufferCreate(nil,fw,fh,kCVPixelFormatType_TwoComponent16Half,attrs as CFDictionary,&p)==kCVReturnSuccess,let p,let io=CVPixelBufferGetIOSurface(p)?.takeUnretainedValue() else{throw NSError(domain:"AppleFG",code:9)}
  let td=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rg16Float,width:fw,height:fh,mipmapped:false);td.storageMode = .shared;td.usage=[.shaderRead]
  guard let t=device.makeTexture(descriptor:td,iosurface:io,plane:0) else{throw NSError(domain:"AppleFG",code:9)};return(p,t)
 }
 private func finish(_ cb:MTLCommandBuffer)throws{cb.commit();cb.waitUntilCompleted();guard cb.status == .completed else{throw cb.error ?? NSError(domain:"AppleFG",code:5)}}
 private func convert(_ input:MTLTexture)throws->(CVPixelBuffer,MTLTexture) {
  let target=try surface(half:true),cb=queue.makeCommandBuffer()!,e=cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(toHalf);e.setTexture(input,index:0);e.setTexture(target.1,index:1);e.dispatchThreads(MTLSize(width:1920,height:1080,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1));e.endEncoding();try finish(cb);return target
 }
 private func frame(_ p:CVPixelBuffer,_ t:Double)->VTFrameProcessorFrame{VTFrameProcessorFrame(buffer:p,presentationTimeStamp:CMTime(seconds:t,preferredTimescale:90000))!}
 private var historyInput: (CVPixelBuffer, Double)?
 func resetHistory() { historyInput = nil }
 func estimateBackward(_ input: CVPixelBuffer, time: Double) throws -> (Int, Int, [Float])? {
  if let flight, !flight.snapshot().0 { resetHistory(); return nil }
  let current = try convert(try inputTexture(input)).0
  let previous = historyInput
  historyInput = (current, time)
  guard let (old, oldTime) = previous, time > oldTime, time - oldTime <= 0.25 else { return nil }
  let forward = try flowBuffer(), backward = try flowBuffer()
  guard let flow = VTFrameProcessorOpticalFlow(forwardFlow: forward.0, backwardFlow: backward.0),
        let parameters = VTOpticalFlowParameters(sourceFrame: frame(old, oldTime), nextFrame: frame(current, time), submissionMode: .random, destinationOpticalFlow: flow) else {
   throw NSError(domain: "NR Motion", code: 1)
  }
  let request = Flight()
  flight = request
  processor.process(parameters: parameters) { [parameters] _, error in
   _ = parameters
   request.complete(error)
  }
  guard request.done.wait(timeout: .now() + 0.08) == .success else { resetHistory(); return nil }
  if let error = request.snapshot().1 { resetHistory(); throw error }
  let pixels = backward.0
  guard CVPixelBufferLockBaseAddress(pixels, .readOnly) == kCVReturnSuccess else {
   throw NSError(domain: "NR Motion", code: 2)
  }
  defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
  let data = CVPixelBufferGetBaseAddress(pixels)!.assumingMemoryBound(to: Float16.self)
  let stride = CVPixelBufferGetBytesPerRow(pixels) / MemoryLayout<Float16>.stride
  var values = [Float]()
  values.reserveCapacity(fw * fh * 2)
  for y in 0..<fh { for x in 0..<(fw * 2) { values.append(Float(data[y * stride + x])) } }
  return (fw, fh, values)
 }
 private func inputTexture(_ input: CVPixelBuffer) throws -> MTLTexture {
  let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
      width: CVPixelBufferGetWidth(input), height: CVPixelBufferGetHeight(input), mipmapped: false)
  descriptor.storageMode = .shared
  descriptor.usage = .shaderRead
  guard let surface = CVPixelBufferGetIOSurface(input)?.takeUnretainedValue(),
        let texture = device.makeTexture(descriptor: descriptor, iosurface: surface, plane: 0) else {
   throw NSError(domain: "NR Motion", code: 3)
  }
  return texture
 }
 func generate(_ base:ProcessedFrame,factor:Int,guides:[CaptureGuide]=[],guided:Bool=false)throws->MultiFrameInterpolator.Result {
  let start=ProcessInfo.processInfo.systemUptime
  func empty(_ diff:Double=0)->MultiFrameInterpolator.Result{.init(frames:[],milliseconds:(ProcessInfo.processInfo.systemUptime-start)*1000,cut:true,difference:diff,validationMaxDelta:-1)}
  if let f=flight,!f.snapshot().0{reset();skip();return empty()}
  let style=try convert(base.texture),raw=guided ? try convert(base.motionGuide?.texture ?? base.texture):style,a=previous,b=Endpoint(frame:base,raw:raw);previous=b
  guard let a else{return empty()}
  let gap=base.sourceSeconds-a.frame.sourceSeconds;guard gap>0,gap<=0.25 else{random=true;return empty()}
  let targets=guided ? CaptureGuideSelection.select(guides,after:a.frame.sourceSeconds,before:base.sourceSeconds,factor:factor):[]
  if guided && (targets.isEmpty || base.motionGuide==nil || a.frame.motionGuide==nil){random=true;return empty()}
  let phases=guided ? targets.map{Float(($0.seconds-a.frame.sourceSeconds)/gap)}:(1..<factor).map{Float($0)/Float(factor)}
  CVPixelBufferLockBaseAddress(a.raw.0,.readOnly);CVPixelBufferLockBaseAddress(raw.0,.readOnly)
  let ap=CVPixelBufferGetBaseAddress(a.raw.0)!.assumingMemoryBound(to:Float16.self),bp=CVPixelBufferGetBaseAddress(raw.0)!.assumingMemoryBound(to:Float16.self),ar=CVPixelBufferGetBytesPerRow(a.raw.0)/2,br=CVPixelBufferGetBytesPerRow(raw.0)/2;var diff=0.0
  for y in stride(from:30,to:1080,by:60){for x in stride(from:30,to:1920,by:60){for c in 0..<3{diff+=Double(abs(Float(ap[y*ar+x*4+c])-Float(bp[y*br+x*4+c])))/Double(18*32*3)}}}
  CVPixelBufferUnlockBaseAddress(raw.0,.readOnly);CVPixelBufferUnlockBaseAddress(a.raw.0,.readOnly)
  if diff>=0.25{random=true;return empty(diff)}
  let f=try flowBuffer(),r=try flowBuffer(),flow=VTFrameProcessorOpticalFlow(forwardFlow:f.0,backwardFlow:r.0)!
  if diff<0.0001 {
   for buffer in [f.0,r.0] {CVPixelBufferLockBaseAddress(buffer,[]);memset(CVPixelBufferGetBaseAddress(buffer)!,0,CVPixelBufferGetBytesPerRow(buffer)*CVPixelBufferGetHeight(buffer));CVPixelBufferUnlockBaseAddress(buffer,[])}
   random=true
  } else {
  guard let params=VTOpticalFlowParameters(sourceFrame:frame(a.raw.0,a.frame.sourceSeconds),nextFrame:frame(raw.0,base.sourceSeconds),submissionMode:random ? .random:.sequential,destinationOpticalFlow:flow) else{throw NSError(domain:"AppleFG",code:10)}
  let request=Flight();flight=request;processor.process(parameters:params,completionHandler:{[params] _,error in _ = params;request.complete(error)})
  guard request.done.wait(timeout:.now()+0.08) == .success else{reset();skip();return empty()}
  if let error=request.snapshot().1{throw error};random=false
  }
  let cb=queue.makeCommandBuffer()!;var result:[ProcessedFrame]=[]
  for (i,phase) in phases.enumerated(){
   let out=try surface(half:false),e=cb.makeComputeCommandEncoder()!;var flags:[UInt32]=[phase.bitPattern,guided ? 1:0]
   e.setComputePipelineState(pack);e.setTexture(style.1,index:0);e.setTexture(out.1,index:1);e.setTexture(raw.1,index:2);e.setTexture(guided ? targets[i].texture:base.texture,index:3);e.setTexture(r.1,index:4);e.setTexture(f.1,index:5)
   flags.withUnsafeMutableBytes{e.setBytes($0.baseAddress!,length:$0.count,index:0)}
   e.dispatchThreads(MTLSize(width:1920,height:1080,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1));e.endEncoding()
   let frame=ProcessedFrame(out.0,out.1,base.sequence,guided ? targets[i].seconds:a.frame.sourceSeconds+gap*Double(phase),0,0);frame.generated=true;frame.captureGuided=guided;frame.phase=Double(phase);frame.pairStart=a.frame.sequence;frame.pairEnd=base.sequence;result.append(frame)
  }
  try finish(cb);withExtendedLifetime((a,b,f,r,style,targets)){}
  return .init(frames:result,milliseconds:(ProcessInfo.processInfo.systemUptime-start)*1000,cut:false,difference:diff,validationMaxDelta:-1)
 }
 func prepare()throws{
  let first=try surface(half:false),second=try surface(half:false)
  for (index,surface) in [first,second].enumerated(){
   CVPixelBufferLockBaseAddress(surface.0,[]);let p=CVPixelBufferGetBaseAddress(surface.0)!.assumingMemoryBound(to:UInt8.self),row=CVPixelBufferGetBytesPerRow(surface.0)
   for y in 0..<1080{for x in 0..<1920{let i=y*row+x*4;p[i]=64;p[i+1]=UInt8(y%251);p[i+2]=UInt8((x+index*4)%251);p[i+3]=255}}
   CVPixelBufferUnlockBaseAddress(surface.0,[])
  }
  let a=ProcessedFrame(first.0,first.1,0,1,0,0),b=ProcessedFrame(second.0,second.1,6,1.1,0,0);a.motionGuide=CaptureGuide(texture:first.1,sequence:0,seconds:1);b.motionGuide=CaptureGuide(texture:second.1,sequence:6,seconds:1.1)
  let guides=(1...5).map{CaptureGuide(texture:first.1,sequence:UInt64($0),seconds:1+Double($0)/60)}
  for _ in 0..<2{reset();_ = try generate(a,factor:6,guides:guides,guided:true);_ = try generate(b,factor:6,guides:guides,guided:true);if let f=flight,!f.snapshot().0{guard f.done.wait(timeout:.now()+5) == .success else{throw NSError(domain:"AppleFG",code:7)}};if let error=flight?.snapshot().1{throw error}}
  stateLock.lock();skipped=0;stateLock.unlock();reset()
 }
 static let shader="""
 #include <metal_stdlib>
 using namespace metal;
 \(guidedAppearanceMetal)
 constexpr sampler s(filter::linear,address::clamp_to_edge);
 kernel void to_half(texture2d<float> a [[texture(0)]],texture2d<half,access::write> b [[texture(1)]],uint2 p [[thread_position_in_grid]]){
  if(p.x>=b.get_width()||p.y>=b.get_height())return;b.write(half4(a.sample(s,(float2(p)+.5f)/float2(b.get_width(),b.get_height()))),p);
 }
 struct Params{float time;uint guided;};
 kernel void anchor_pack(texture2d<half> style [[texture(0)]],texture2d<float,access::write> out [[texture(1)]],texture2d<half> raw [[texture(2)]],texture2d<float> actual [[texture(3)]],texture2d<half> backward [[texture(4)]],texture2d<half> forward [[texture(5)]],constant Params &q [[buffer(0)]],uint2 p [[thread_position_in_grid]]){
  if(p.x>=out.get_width()||p.y>=out.get_height())return;
  float2 size=float2(out.get_width(),out.get_height()),grid=float2(backward.get_width(),backward.get_height()),here=(float2(p)+.5f)/size,uv=here;
  // Calibrated native flow values are displacement in flow-grid pixels.
  for(uint k=0;k<4;k++)uv=here-(1.f-q.time)*float2(backward.sample(s,uv).xy)/grid;
  float3 styled=float3(style.sample(s,uv).rgb),rgb=styled;
  if(q.guided){
   float3 guide=actual.sample(s,here).rgb,reference=float3(raw.sample(s,uv).rgb);float error=0;
   for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){float2 d=float2(x,y)*2.f/size;float3 e=float3(raw.sample(s,uv+d).rgb)-actual.sample(s,here+d).rgb;error+=dot(e,e)/9.f;}
   float2 back=float2(backward.sample(s,uv).xy),oldUV=uv+back/grid,cycle=back+float2(forward.sample(s,oldUV).xy),inverse=(uv+(1.f-q.time)*back/grid-here)*size;
   float2 dx=float2(backward.sample(s,uv+float2(1,0)/grid).xy)-back,dy=float2(backward.sample(s,uv+float2(0,1)/grid).xy)-back,jx=float2(1,0)+(1.f-q.time)*dx,jy=float2(0,1)+(1.f-q.time)*dy;
   float det=jx.x*jy.y-jx.y*jy.x;
   float confidence=exp(-error/.0025f)*exp(-dot(cycle,cycle))*exp(-dot(inverse,inverse)/4.f)*smoothstep(.1f,.4f,det)*(1.f-smoothstep(3.f,8.f,det));
   if(any(uv<0.f)||any(uv>1.f)||any(oldUV<0.f)||any(oldUV>1.f))confidence=0;
   float3 rawMean=reference*2.f,styledMean=styled*2.f;
   for(int axis=0;axis<2;axis++) {
    float2 offset=(axis==0 ? float2(6,0):float2(0,6))/size;
    rawMean+=float3(raw.sample(s,uv+offset).rgb)+float3(raw.sample(s,uv-offset).rgb);
    styledMean+=float3(style.sample(s,uv+offset).rgb)+float3(style.sample(s,uv-offset).rgb);
   }
   rgb=apply_guided_appearance(guide,rawMean/6.f,styledMean/6.f,confidence);
  }
  out.write(float4(round(clamp(rgb,0.f,1.f)*255.f)/255.f,1.f),p);
 }
 """
}
/// Session/model startup must not block the serial NR worker or the UI.
final class AppleFGService {
 private let lock=NSLock()
 private var engine:AnyObject?,failure="",loading=true
 init(device:MTLDevice) {
  DispatchQueue.global(qos:.userInitiated).async { [self] in
   do {
    if #available(macOS 15.4,*) {
     let result=try AppleFrameInterpolator(device:device)
     try result.prepare()
     lock.lock();engine=result;loading=false;lock.unlock()
    } else {throw NSError(domain:"AppleFG",code:1)}
   } catch {lock.lock();failure=String(describing:error);loading=false;lock.unlock()}
  }
 }
 func ready()->AnyObject? {lock.lock();defer{lock.unlock()};return engine}
 func status()->(String,String) {lock.lock();defer{lock.unlock()};return(loading ? "warming-base-only" : engine == nil ? "failed-base-only" : "ready",failure)}
}
