import Foundation
import CoreVideo
import Metal
import MetalPerformanceShaders
import DLSSMLX
import MLX

/// Learned phase batches, followed by direct 1080p texture warp/packing.
/// No full-resolution RGB phase batches or generated-frame feedback.
final class MultiFrameInterpolator {
 struct Endpoint {let frame:ProcessedFrame,low:MLXArray}
 struct Result {let frames:[ProcessedFrame],milliseconds:Double,cut:Bool,difference:Double,validationMaxDelta:Int}
 private let device:MTLDevice,queue:MTLCommandQueue
 private let scaler:MPSImageLanczosScale,unpack:MTLComputePipelineState,warp:MTLComputePipelineState,phaseWarp:MTLComputePipelineState
 private let referencePack:MLXFast.MLXFastKernel
 private let network:FrameGenerator
 private let singleTrajectory=ProcessInfo.processInfo.environment["DLSS_MFG_LEGACY"] != "1"
 private var previous:Endpoint?
 private var fh=360
 private var fw:Int {fh*16/9}
 private func outputScale(for field:MLXArray)throws->Float {
  guard field.ndim==4,field.dim(1)*2==fh,field.dim(2)*2==fw,field.dim(3)==8 else {
   throw NSError(domain:"MFG",code:11,userInfo:[NSLocalizedDescriptionKey:"Unexpected MLX motion field shape: \(field.shape)"])
  }
  // The exported vectors are in half-resolution field pixels.
  return 1080/Float(field.dim(1))
 }
 private var validated:Set<String>=[]
 func setResolution(height:Int) {if height != fh {fh=height;reset();validated=[]}}
 init(device:MTLDevice,path:String)throws {
  self.device=device;scaler=MPSImageLanczosScale(device:device)
  guard let queue=device.makeCommandQueue() else {throw NSError(domain:"MFG",code:1)}
  self.queue=queue;network=try FrameGenerator(weightsURL:URL(fileURLWithPath:path),precision:.float16)
  let library=try device.makeLibrary(source:"""
  #include <metal_stdlib>
  using namespace metal;
  \(guidedAppearanceMetal)
  kernel void unpack(texture2d<float,access::read> input [[texture(0)]],device float *rgb [[buffer(0)]],uint2 p [[thread_position_in_grid]]) {
   if(p.x>=input.get_width() || p.y>=input.get_height())return;
   uint i=(p.y*input.get_width()+p.x)*3;float3 v=input.read(p).rgb;for(uint c=0;c<3;c++)rgb[i+c]=v[c];
  }
  float3 source_at(texture2d<float,access::read> input,float2 p) {
   int2 b=int2(floor(p)),limit=int2(input.get_width(),input.get_height())-1;float2 t=p-float2(b);
   float3 a=input.read(uint2(clamp(b,int2(0),limit))).rgb;
   float3 c=input.read(uint2(clamp(b+int2(1,0),int2(0),limit))).rgb;
   float3 d=input.read(uint2(clamp(b+int2(0,1),int2(0),limit))).rgb;
   float3 e=input.read(uint2(clamp(b+int2(1,1),int2(0),limit))).rgb;
   return (1-t.x)*(1-t.y)*a+t.x*(1-t.y)*c+(1-t.x)*t.y*d+t.x*t.y*e;
  }
  struct Params {uint h,w,phase;float scale;float time;uint guided;};
  kernel void legacy_warp_pack(texture2d<float,access::read> a [[texture(0)]],texture2d<float,access::read> b [[texture(1)]],texture2d<float,access::write> out [[texture(2)]],device const float *all [[buffer(0)]],constant Params &q [[buffer(1)]],uint2 p [[thread_position_in_grid]]) {
   if(p.x>=1920 || p.y>=1080)return;
   device const float *co=all+q.phase*q.h*q.w*8;
   float2 uv=min(float2(p)/q.scale,float2(q.w-1,q.h-1)),t=fract(uv);uint2 i=uint2(floor(uv)),j=min(i+1,uint2(q.w-1,q.h-1));
   uint offsets[4]={(i.y*q.w+i.x)*8,(i.y*q.w+j.x)*8,(j.y*q.w+i.x)*8,(j.y*q.w+j.x)*8};
   float up[5];
   for(uint c=0;c<5;c++) {
    float v0=co[offsets[0]+c],v1=co[offsets[1]+c],v2=co[offsets[2]+c],v3=co[offsets[3]+c];
    if(c==4){v0=1/(1+exp(-v0));v1=1/(1+exp(-v1));v2=1/(1+exp(-v2));v3=1/(1+exp(-v3));}
    up[c]=(1-t.x)*(1-t.y)*v0+t.x*(1-t.y)*v1+(1-t.x)*t.y*v2+t.x*t.y*v3;
   }
   float3 rgb=up[4]*source_at(a,float2(p)+float2(up[0],up[1])*q.scale)+(1-up[4])*source_at(b,float2(p)+float2(up[2],up[3])*q.scale);
   // Quantize explicitly to match the established RGB8 reference packing.
   out.write(float4(round(clamp(rgb,0.f,1.f)*255.f)/255.f,1.f),p);
  }
  float4 trajectory_field(device const float *co,uint w,uint h,float2 uv) {
   uv=clamp(uv,float2(0),float2(w-1,h-1));uint2 i=uint2(floor(uv)),j=min(i+1,uint2(w-1,h-1));float2 t=fract(uv);
   float4 v[4];uint offsets[4]={(i.y*w+i.x)*8,(i.y*w+j.x)*8,(j.y*w+i.x)*8,(j.y*w+j.x)*8};
   for(uint n=0;n<4;n++)v[n]=float4(co[offsets[n]],co[offsets[n]+1],co[offsets[n]+2],co[offsets[n]+3]);
   return mix(mix(v[0],v[1],t.x),mix(v[2],v[3],t.x),t.y);
  }
  float visibility_logit(device const float *field,uint w,uint h,float2 uv) {
   uv=clamp(uv,float2(0),float2(w-1,h-1));
   uint2 a=uint2(floor(uv)), b=min(a+1,uint2(w-1,h-1));
   float2 t=fract(uv);
   return mix(mix(field[(a.y*w+a.x)*8+4],field[(a.y*w+b.x)*8+4],t.x),
              mix(field[(b.y*w+a.x)*8+4],field[(b.y*w+b.x)*8+4],t.x),t.y);
  }
  kernel void phase_pack(texture2d<float,access::read> a [[texture(0)]],
                         texture2d<float,access::read> b [[texture(1)]],
                         texture2d<float,access::write> out [[texture(2)]],
                         texture2d<float> rawA [[texture(3)]],
                         texture2d<float> rawB [[texture(4)]],
                         device const float *packedFields [[buffer(0)]],
                         constant Params &q [[buffer(1)]],uint2 p [[thread_position_in_grid]]) {
   if(p.x>=out.get_width() || p.y>=out.get_height()) return;
   device const float *field=packedFields+q.phase*q.h*q.w*8;
   float2 pixel=float2(p), uv=pixel/q.scale;
   float4 f=trajectory_field(field,q.w,q.h,uv)*q.scale;
   float4 dx=(trajectory_field(field,q.w,q.h,uv+float2(2.f/q.scale,0))*q.scale-f)*.5f;
   float4 dy=(trajectory_field(field,q.w,q.h,uv+float2(0,2.f/q.scale))*q.scale-f)*.5f;
   float2 pa=pixel+f.xy,pb=pixel+f.zw,last=float2(out.get_width()-1,out.get_height()-1);
   float detA=(1.f+dx.x)*(1.f+dy.y)-dx.y*dy.x;
   float detB=(1.f+dx.z)*(1.f+dy.w)-dx.w*dy.z;
   bool validA=all(pa>=0.f)&&all(pa<=last)&&detA>.2f&&detA<5.f;
   bool validB=all(pb>=0.f)&&all(pb<=last)&&detB>.2f&&detB<5.f;
   float weight=1.f/(1.f+exp(-visibility_logit(field,q.w,q.h,uv)));
   // At a motion boundary, interpolating vectors from two surfaces can
   // sample neither surface. Pick one measured hypothesis using raw color
   // agreement, so differences introduced by NR cannot steer geometry.
   float jump=max(length(dx.xy),max(length(dx.zw),max(length(dy.xy),length(dy.zw))));
   if (jump>2.f || !validA || !validB) {
    constexpr sampler linearClamp(filter::linear,address::clamp_to_edge);
    float2 cell=floor(uv), size=last+1.f;
    float best=INFINITY, bestWeight=weight;
    float2 bestA=pa,bestB=pb;
    for (uint k=0;k<4;k++) {
     float2 tap=min(cell+float2(k&1,k>>1),float2(q.w-1,q.h-1));
     uint i=(uint(tap.y)*q.w+uint(tap.x))*8;
     float2 ca=pixel+float2(field[i],field[i+1])*q.scale;
     float2 cb=pixel+float2(field[i+2],field[i+3])*q.scale;
     if (any(ca<0.f)||any(ca>last)||any(cb<0.f)||any(cb>last)) continue;
     float3 va=rawA.sample(linearClamp,(ca+.5f)/size).rgb;
     float3 vb=rawB.sample(linearClamp,(cb+.5f)/size).rgb;
     float2 distance=tap-uv;
     float cost=length(va-vb)+.045f*dot(distance,distance);
     if (cost<best) {
      best=cost; bestA=ca; bestB=cb;
      bestWeight=1.f/(1.f+exp(-field[i+4]));
     }
    }
    if (best<.12f) {pa=bestA; pb=bestB; weight=bestWeight; validA=true; validB=true;}
    else if (jump>2.f) {validA=false; validB=false;}
   }
   float3 rgb;
   if (validA && validB) {
    float3 ca=source_at(a,pa),cb=source_at(b,pb);
    // Do not crossfade incompatible surfaces at an occlusion boundary.
    rgb=length(ca-cb)<.035f ? mix(cb,ca,weight) : (weight>=.5f ? ca : cb);
   } else if (validA) rgb=source_at(a,pa);
   else if (validB) rgb=source_at(b,pb);
   else rgb=q.time<.5f ? source_at(a,pixel) : source_at(b,pixel);
   out.write(float4(clamp(rgb,0.f,1.f),1.f),p);
  }
  kernel void trajectory_pack(texture2d<float,access::read> a [[texture(0)]],texture2d<float,access::read> b [[texture(1)]],texture2d<float,access::write> out [[texture(2)]],texture2d<float> rawA [[texture(3)]],texture2d<float> rawB [[texture(4)]],texture2d<float> actual [[texture(5)]],device const float *co [[buffer(0)]],constant Params &q [[buffer(1)]],texture2d<float> proxyRaw [[texture(6)]],texture2d<float> proxyStyled [[texture(7)]],texture2d<float> earlierRaw [[texture(8)]],texture2d<float> earlierStyled [[texture(9)]],texture2d<float> earlierGuide [[texture(10)]],constant float &gain [[buffer(2)]],constant float4 &hud [[buffer(3)]],constant float3 &colorControls [[buffer(4)]],uint2 p [[thread_position_in_grid]]) {
   if(p.x>=1920 || p.y>=1080)return;
   float2 pixel=float2(p),mid=pixel;
   for(uint k=0;k<2;k++) {float4 f=trajectory_field(co,q.w,q.h,mid/q.scale)*q.scale;mid=pixel-mix(f.xy,f.zw,q.time);}
   float4 f=trajectory_field(co,q.w,q.h,mid/q.scale)*q.scale;
   // The endpoint choice is uniform across the entire frame. There is no
   // spatial confidence switch, RGB crossfade or pair-rejection rate change.
   float3 rgb=q.time<.5f ? source_at(a,mid+f.xy) : source_at(b,mid+f.zw);
   if(q.guided) {
    constexpr sampler s(filter::linear,address::clamp_to_edge);
    float2 size=float2(1920,1080),ub=(mid+f.zw+.5f)/size,here=(float2(p)+.5f)/size;
    float3 guide=actual.sample(s,here).rgb,raw=rawB.sample(s,ub).rgb;
    float2 inverse=mid+mix(f.xy,f.zw,q.time)-pixel;
    float4 fx=trajectory_field(co,q.w,q.h,(mid+float2(2,0))/q.scale)*q.scale,fy=trajectory_field(co,q.w,q.h,(mid+float2(0,2))/q.scale)*q.scale;
    float2 jx=float2(1,0)+(mix(fx.xy-f.xy,fx.zw-f.zw,q.time))*.5f,jy=float2(0,1)+(mix(fy.xy-f.xy,fy.zw-f.zw,q.time))*.5f;
    float determinant=jx.x*jy.y-jx.y*jy.x;
    float confidence=exp(-dot(inverse,inverse)/4.f)*smoothstep(.1f,.4f,determinant)*(1.f-smoothstep(3.f,8.f,determinant));
    if(any(mid<0.f)||any(mid>size-1.f))confidence=0;
    if (gain >= 0.f) {
    bool protect=all(here >= hud.xy) && all(here < hud.zw);
    float2 ua=(mid+f.xy+.5f)/size;
    float3 matchA=register_guide(earlierGuide,actual,ua,here,size);
    float3 matchB=register_guide(rawB,actual,ub,here,size);
    float trustA=confidence*matchA.z;
    confidence*=matchB.z;
    CorrectionSample before=sample_correction(earlierStyled,earlierRaw,matchA.xy,guide);
    CorrectionSample after=sample_correction(proxyStyled,proxyRaw,matchB.xy,guide);
    rgb=protect ? guide : compose_transported_detail(guide,before,after,q.time,trustA,confidence,gain,colorControls);
   } else {
    float error=0;
    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
     float2 offset=float2(x,y)*2.f/size;
     float3 d=rawB.sample(s,ub+offset).rgb-actual.sample(s,here+offset).rgb;error+=dot(d,d)/9.f;
    }
    confidence*=exp(-error/.0025f);
    if(any(ub<0.f)||any(ub>1.f))confidence=0.f;
    float3 rawMean=raw*2.f,styledMean=source_at(b,mid+f.zw)*2.f;
    for(int axis=0;axis<2;axis++) {
     float2 offset=axis==0 ? float2(6,0):float2(0,6);
     rawMean+=rawB.sample(s,ub+offset/size).rgb+rawB.sample(s,ub-offset/size).rgb;
     styledMean+=source_at(b,mid+f.zw+offset)+source_at(b,mid+f.zw-offset);
    }
    rgb=apply_guided_appearance(guide,rawMean/6.f,styledMean/6.f,confidence);
   }
   } else {
    float2 pa=mid+f.xy,pb=mid+f.zw;
    float2 last=float2(1919,1079);
    float2 inverse=mid+mix(f.xy,f.zw,q.time)-pixel;
    float4 fx=trajectory_field(co,q.w,q.h,(mid+float2(2,0))/q.scale)*q.scale-f;
    float4 fy=trajectory_field(co,q.w,q.h,(mid+float2(0,2))/q.scale)*q.scale-f;
    float detA=(1.f+.5f*fx.x)*(1.f+.5f*fy.y)-.25f*fx.y*fy.x;
    float detB=(1.f+.5f*fx.z)*(1.f+.5f*fy.w)-.25f*fx.w*fy.z;
    bool validA=all(pa>=0.f)&&all(pa<=last)&&detA>.2f&&detA<5.f;
    bool validB=all(pb>=0.f)&&all(pb<=last)&&detB>.2f&&detB<5.f;
    if (dot(inverse,inverse)>4.f || (!validA && !validB))
     rgb=q.time<.5f ? source_at(a,pixel) : source_at(b,pixel);
    else if (validA && (!validB || q.time<.5f)) rgb=source_at(a,pa);
    else rgb=source_at(b,pb);
   }
   out.write(float4(round(clamp(rgb,0.f,1.f)*255.f)/255.f,1.f),p);
  }
  """,options:nil)
  unpack=try device.makeComputePipelineState(function:library.makeFunction(name:"unpack")!)
  warp=try device.makeComputePipelineState(function:library.makeFunction(name:singleTrajectory ? "trajectory_pack" : "legacy_warp_pack")!)
  phaseWarp=try device.makeComputePipelineState(function:library.makeFunction(name:"phase_pack")!)
  referencePack=MLXFast.metalKernel(name:"mfg_reference_pack",inputNames:["rgb"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;if(p>=1920*1080)return;
  for(uint c=0;c<3;c++)out[p*4+2-c]=uchar(clamp(round(float(rgb[p*3+c])*255.f),0.f,255.f));out[p*4+3]=255;
  """)
 }
 func reset() {previous=nil;validated=[]}
 private func finish(_ cb:MTLCommandBuffer)throws {cb.commit();cb.waitUntilCompleted();guard cb.status == .completed else {throw cb.error ?? NSError(domain:"MFG",code:2)}}
 private func encodeRGB(_ input:MTLTexture,_ buffer:MTLBuffer,_ cb:MTLCommandBuffer)throws {
  guard let enc=cb.makeComputeCommandEncoder() else {throw NSError(domain:"MFG",code:3)}
  enc.setComputePipelineState(unpack);enc.setTexture(input,index:0);enc.setBuffer(buffer,offset:0,index:0)
  enc.dispatchThreads(MTLSize(width:input.width,height:input.height,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1));enc.endEncoding()
 }
 private func endpoint(_ frame:ProcessedFrame)throws->Endpoint {
  guard frame.texture.width==1920,frame.texture.height==1080,let cb=queue.makeCommandBuffer(),let low=device.makeBuffer(length:fw*fh*3*4,options:.storageModeShared) else {throw NSError(domain:"MFG",code:3)}
  let td=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.bgra8Unorm,width:fw,height:fh,mipmapped:false);td.storageMode = .shared;td.usage=[.shaderRead,.shaderWrite]
  guard let proxy=device.makeTexture(descriptor:td) else {throw NSError(domain:"MFG",code:3)}
  scaler.encode(commandBuffer:cb,sourceTexture:frame.motionGuide?.texture ?? frame.texture,destinationTexture:proxy);try encodeRGB(proxy,low,cb);try finish(cb)
  let l=MLXArray(rawPointer:low.contents(),[1,fh,fw,3],dtype:.float32){_ = low}.asType(.float16);eval(l)
  return Endpoint(frame:frame,low:l)
 }
 private func rgb(_ frame:ProcessedFrame)throws->MLXArray {
  guard let cb=queue.makeCommandBuffer(),let full=device.makeBuffer(length:1920*1080*3*4,options:.storageModeShared) else {throw NSError(domain:"MFG",code:3)}
  try encodeRGB(frame.texture,full,cb);try finish(cb)
  return MLXArray(rawPointer:full.contents(),[1,1080,1920,3],dtype:.float32){_ = full}
 }
 private func outputs(_ coarse:MLXArray,a:ProcessedFrame,b:ProcessedFrame,phases:[Float],targets:[CaptureGuide]=[],phaseSpecific:Bool=false)throws->[ProcessedFrame] {
  let scale=try outputScale(for:coarse)
  let storage=contiguous(coarse.asType(.float32));eval(storage)
  guard let buffer=storage.asMTLBuffer(device:device,noCopy:true),let cb=queue.makeCommandBuffer() else {throw NSError(domain:"MFG",code:4)}
  var result:[ProcessedFrame]=[]
  for (index,phase) in phases.enumerated() {
   var pixels:CVPixelBuffer?
   let attrs:[CFString:Any]=[kCVPixelBufferIOSurfacePropertiesKey:[:],kCVPixelBufferMetalCompatibilityKey:true]
   guard CVPixelBufferCreate(nil,1920,1080,kCVPixelFormatType_32BGRA,attrs as CFDictionary,&pixels)==kCVReturnSuccess,let pixels,let surface=CVPixelBufferGetIOSurface(pixels)?.takeUnretainedValue() else {throw NSError(domain:"MFG",code:4)}
   let td=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.bgra8Unorm,width:1920,height:1080,mipmapped:false);td.storageMode = .shared;td.usage=[.shaderRead,.shaderWrite]
   guard let texture=device.makeTexture(descriptor:td,iosurface:surface,plane:0),let enc=cb.makeComputeCommandEncoder() else {throw NSError(domain:"MFG",code:4)}
   var parameters:[UInt32]=[UInt32(storage.dim(1)),UInt32(storage.dim(2)),UInt32(index),scale.bitPattern,phase.bitPattern,targets.isEmpty ? 0:1]
   enc.setComputePipelineState(phaseSpecific ? phaseWarp : warp);enc.setTexture(a.texture,index:0);enc.setTexture(b.texture,index:1);enc.setTexture(texture,index:2);enc.setBuffer(buffer,offset:0,index:0)
   if singleTrajectory {enc.setTexture(a.motionGuide?.texture ?? a.texture,index:3);enc.setTexture(b.motionGuide?.texture ?? b.texture,index:4)}
   if singleTrajectory && !phaseSpecific {enc.setTexture(targets.isEmpty ? a.texture : targets[index].texture,index:5)}
   if singleTrajectory && !phaseSpecific {
    let proxy=b.appearanceProxy
    let earlier=a.appearanceProxy
    var gain=earlier != nil ? (proxy?.gain ?? -1) : -1
    var hud=proxy?.hud ?? SIMD4<Float>(repeating:0)
    enc.setTexture(proxy?.source.1 ?? b.texture,index:6)
    enc.setTexture(proxy?.correction ?? b.texture,index:7)
    enc.setTexture(earlier?.source.1 ?? a.texture,index:8)
    enc.setTexture(earlier?.correction ?? a.texture,index:9)
    enc.setTexture(a.motionGuide?.texture ?? a.texture,index:10)
    enc.setBytes(&gain,length:MemoryLayout<Float>.stride,index:2)
    enc.setBytes(&hud,length:MemoryLayout<SIMD4<Float>>.stride,index:3)
    var colorControls=proxy?.colorControls ?? SIMD3<Float>(repeating:1)
    enc.setBytes(&colorControls,length:MemoryLayout<SIMD3<Float>>.stride,index:4)
   }
   parameters.withUnsafeMutableBytes{enc.setBytes($0.baseAddress!,length:$0.count,index:1)}
   enc.dispatchThreads(MTLSize(width:1920,height:1080,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1));enc.endEncoding()
   let frame=ProcessedFrame(pixels,texture,b.sequence,targets.isEmpty ? a.sourceSeconds+(b.sourceSeconds-a.sourceSeconds)*Double(phase) : targets[index].seconds,0,0)
   frame.captureGuided = !targets.isEmpty;frame.generated=true;frame.phase=Double(phase);frame.pairStart=a.sequence;frame.pairEnd=b.sequence;result.append(frame)
  }
  try finish(cb);withExtendedLifetime((storage,buffer)){}
  return result
 }
 private func validate(_ actual:ProcessedFrame,coarse:MLXArray,a:ProcessedFrame,b:ProcessedFrame)throws->Int {
  let reference=FrameGenerator.compose(try rgb(a),try rgb(b),coarse:coarse[0..<1],scale:try outputScale(for:coarse),sigmoidMask:true)
  let packed=referencePack([reference],grid:(1920*1080,1,1),threadGroup:(256,1,1),outputShapes:[[1080,1920,4]],outputDTypes:[.uint8])[0]
  let bytes=packed.asArray(UInt8.self)
  CVPixelBufferLockBaseAddress(actual.pixels,.readOnly);defer{CVPixelBufferUnlockBaseAddress(actual.pixels,.readOnly)}
  let base=CVPixelBufferGetBaseAddress(actual.pixels)!.assumingMemoryBound(to:UInt8.self),stride=CVPixelBufferGetBytesPerRow(actual.pixels)
  var maximum=0
  for y in 0..<1080 {for x in 0..<(1920*4) {maximum=max(maximum,abs(Int(base[y*stride+x])-Int(bytes[y*1920*4+x])))}}
  guard maximum<=1 else {throw NSError(domain:"MFG",code:7,userInfo:[NSLocalizedDescriptionKey:"Direct warp parity exceeded one code value: \(maximum)"])}
  return maximum
 }
 func generate(_ base:ProcessedFrame,factor:Int,guides:[CaptureGuide]=[],guided:Bool=false)throws->Result {
  let start=ProcessInfo.processInfo.systemUptime
  if guided && !singleTrajectory {throw NSError(domain:"MFG",code:10,userInfo:[NSLocalizedDescriptionKey:"Capture guidance requires the single-trajectory compositor"])}
  let b=try endpoint(base);let a=previous;previous=b
  guard let a else {return Result(frames:[],milliseconds:(ProcessInfo.processInfo.systemUptime-start)*1000,cut:true,difference:0,validationMaxDelta:-1)}
  let gap=base.sourceSeconds-a.frame.sourceSeconds
  let difference=Double(mean(abs(a.low.asType(.float32)-b.low.asType(.float32))).item(Float.self))
  guard difference.isFinite else {throw NSError(domain:"MFG",code:6,userInfo:[NSLocalizedDescriptionKey:"Nonfinite pair-difference metric"])}
  guard gap>0,gap<=0.25,difference<0.25 else {return Result(frames:[],milliseconds:(ProcessInfo.processInfo.systemUptime-start)*1000,cut:true,difference:difference,validationMaxDelta:-1)}
  let targets=guided ? CaptureGuideSelection.select(guides,after:a.frame.sourceSeconds,before:base.sourceSeconds,factor:factor) : []
  if guided && (targets.isEmpty || a.frame.motionGuide==nil || base.motionGuide==nil) {return .init(frames:[],milliseconds:(ProcessInfo.processInfo.systemUptime-start)*1000,cut:true,difference:difference,validationMaxDelta:-1)}
  let phases=guided ? targets.map{Float(($0.seconds-a.frame.sourceSeconds)/gap)} : (1..<factor).map{Float($0)/Float(factor)}
  let count=phases.count
  let phaseSpecific=singleTrajectory && !guided
  let midpoint=singleTrajectory && guided
  let batch=midpoint ? 1 : count
  var coarse=network.synthesize(broadcast(a.low,to:[batch,fh,fw,3]),broadcast(b.low,to:[batch,fh,fw,3]),phases:MLXArray(midpoint ? [Float(0.5)] : phases));eval(coarse)
  let activeHeight=fh/2,activeWidth=fw/2
  let paddedHeight=(activeHeight+15)/16*16,paddedWidth=(activeWidth+15)/16*16
  guard coarse.shape == [batch,paddedHeight,paddedWidth,8] else {
   throw NSError(domain:"MFG",code:11,userInfo:[NSLocalizedDescriptionKey:"Unexpected padded MLX motion field shape: \(coarse.shape)"])
  }
  // The network appends tile padding on the bottom and right. Crop before
  // packing so neither coordinate mapping nor bilinear taps use that region.
  coarse=coarse[0...,0..<activeHeight,0..<activeWidth,0...]
  if difference<0.0001 {coarse=MLXArray.zeros(like:coarse);eval(coarse)}
  guard all(isFinite(coarse)).item(Bool.self) else {throw NSError(domain:"MFG",code:5,userInfo:[NSLocalizedDescriptionKey:"Nonfinite interpolation field"])}
  let frames=try outputs(coarse,a:a.frame,b:base,phases:phases,targets:targets,phaseSpecific:phaseSpecific)
  if phaseSpecific {base.stages["mfgPhasePredictions"]=Double(batch)}
  var delta = -1
  let key="\(fh)-\(factor)"
  if !guided && !singleTrajectory && !validated.contains(key) && ProcessInfo.processInfo.environment["DLSS_MFG_VALIDATE"]=="1" {
   delta=try validate(frames[0],coarse:coarse,a:a.frame,b:base)
   if count>1 {delta=max(delta,try validate(frames[count-1],coarse:coarse[(count-1)..<count],a:a.frame,b:base))}
   validated.insert(key)
  }
  return Result(frames:frames,milliseconds:(ProcessInfo.processInfo.systemUptime-start)*1000,cut:false,difference:difference,validationMaxDelta:delta)
 }
}
