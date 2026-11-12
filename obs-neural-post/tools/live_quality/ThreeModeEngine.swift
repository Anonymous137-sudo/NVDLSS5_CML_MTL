import Foundation
import CoreVideo
import Metal
import MetalPerformanceShaders
import DLSSCore
import DLSSMLX
import MLX

/// One capture worker owns all MLX arrays. Each mode consumes the current
/// HDMI frame. Pass2 consumes pass1 RGB, with fresh RGB features and unchanged
/// noise/style controls. Temporal mode keeps separate history for each pass.
final class ThreeModeEngine {
 let mode:String
 private var controls:NRConfiguration
 func configure(_ value:NRConfiguration) {
  if controls != value {controls=value;template=nil;previousRaw=nil;passHistory=[];motion=nil;correctionHistory=nil}
 }
 private var lowResolution:Bool {mode.hasSuffix("-480")}
 var label:String {mode == "71x2-480" ? "480p ·71 →71 ·1080p" : mode == "31then71-480" ? "480p ·31 →71 ·1080p" : (mode == "71" || mode == "71-480") ? "71-block · single pass" : mode == "31" ? "31-block Small · single pass" : "31-block Small · double pass"}
 private let device:MTLDevice,queue:MTLCommandQueue,sr:CompactSR
 private let model71:MLXFullFrameRenderer?,model31:MLXFullFrameRenderer?
 private let unpack:MTLComputePipelineState,rgbBuffer:MTLBuffer
 private let composeAppearance, extractCorrection, stabilizeCorrection, mfgCorrection: MTLComputePipelineState
 private var correctionTexture: MTLTexture?
 private var stableTexture: MTLTexture?
 private var correctionHistory: (field: MTLTexture, proxy: (CVPixelBuffer, MTLTexture), pts: Double)?

 private let featuresKernel:MLXFast.MLXFastKernel,pack:MLXFast.MLXFastKernel,pack720:MLXFast.MLXFastKernel
 private let screenColor: MLXFast.MLXFastKernel
 private let alignHistory: MLXFast.MLXFastKernel
 private var motion: AnyObject?
 private var previousRaw: MLXArray?
 private var passHistory: [MLXArray] = []
 private let packRGB:MLXFast.MLXFastKernel
 private let geometry:NeuralRenderingNetworkGeometry
 private let lw:Int,lh:Int,nw:Int,nh:Int
 private let scale:MPSImageLanczosScale,workingTexture:MTLTexture
 private var template:MLXArray?,frames=0
 private let compareEvery:Int
 init(device:MTLDevice,srURL:URL,configuration:NRConfiguration?=nil)throws {
  self.device=device
  func arg(_ prefix:String)->String? {CommandLine.arguments.first(where:{$0.hasPrefix(prefix)}).map{String($0.dropFirst(prefix.count))}}
  mode=configuration?.mode ?? arg("--experiment-mode=") ?? "71"
  controls=configuration ?? NRConfiguration()
  let lowResolution=mode.hasSuffix("-480")
  lw=lowResolution ? 854 : 1280;lh=lowResolution ? 480 : 720;nw=lowResolution ? 896 : 1280;nh=lowResolution ? 512 : 768
  scale=MPSImageLanczosScale(device:device)
  let td=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba32Float,width:lw,height:lh,mipmapped:false);td.storageMode = .shared;td.usage=[.shaderRead,.shaderWrite]
  guard ["71","31","31x2","31then71-480","71x2-480","31-480","71-480","31x2-480"].contains(mode),let texture=device.makeTexture(descriptor:td),let queue=device.makeCommandQueue(),let buffer=device.makeBuffer(length:lw*lh*3*4,options:.storageModeShared) else {throw NSError(domain:"ThreeMode",code:1)}
  self.queue=queue;rgbBuffer=buffer;workingTexture=texture
  compareEvery=lowResolution ? 0 : (Int(arg("--compare-every=") ?? "0") ?? 0)
  let ht=configuration?.model71 ?? arg("--model-71=") ?? ""
  let small=configuration?.model31 ?? arg("--model-31=") ?? ""
  model71=try (mode == "71" || ["31then71-480","71x2-480","71-480"].contains(mode) || compareEvery>0) ? MLXFullFrameRenderer(packageURL:URL(fileURLWithPath:ht),networkWidth:nw,networkHeight:nh) : nil
  model31=try ((mode != "71" && mode != "71x2-480" && mode != "71-480") || compareEvery>0) ? MLXFullFrameRenderer(packageURL:URL(fileURLWithPath:small),networkWidth:nw,networkHeight:nh) : nil
  guard model71 == nil || model71!.blockCount==71,model31 == nil || model31!.blockCount==31 else {throw NSError(domain:"ThreeMode",code:2,userInfo:[NSLocalizedDescriptionKey:"Checkpoint topology does not match selected mode"])}
  sr=try CompactSR(url:srURL)
  geometry=try NeuralRenderingNetworkGeometry(outputWidth:lw,outputHeight:lh,networkWidth:nw,networkHeight:nh)
  let lib=try device.makeLibrary(source:"""
  #include <metal_stdlib>
  using namespace metal;
  kernel void unpack(texture2d<float,access::read> input [[texture(0)]],device float* rgb [[buffer(0)]],uint2 p [[thread_position_in_grid]]) {
   if(p.x>=input.get_width() || p.y>=input.get_height())return;float3 v=clamp(input.read(p).rgb,0.f,1.f);uint i=(p.y*input.get_width()+p.x)*3;
   for(uint c=0;c<3;c++)rgb[i+c]=v[c];
  }
  \(guidedAppearanceMetal)
  kernel void extract_correction(texture2d<float,access::read> styled [[texture(0)]],
                                 texture2d<float,access::read> proxy [[texture(1)]],
                                 texture2d<half,access::write> field [[texture(2)]],
                                 constant float3 &controls [[buffer(0)]],
                                 uint2 p [[thread_position_in_grid]]) {
   if (p.x>=field.get_width() || p.y>=field.get_height()) return;
   float3 source=appearance_decode(proxy.read(p).rgb);
   float3 model=appearance_decode(styled.read(p).rgb);
   if (controls.z!=1.f) model=model*controls.z/(1.f+(controls.z-1.f)*model);
   float3 delta=perceptual_correction(source,model,controls.x,controls.y);
   field.write(half4(half3(delta),0),p);
  }
  // Port of Low Frequency anti-flicker in OBS-DLSS5-Neural-Filter data/neural.hlsl.
  kernel void stabilize_correction(texture2d<half> current [[texture(0)]],
                                   texture2d<half> history [[texture(1)]],
                                   texture2d<float> proxy [[texture(2)]],
                                   texture2d<float> previous [[texture(3)]],
                                   texture2d<half,access::write> output [[texture(4)]],
                                   device const float2 *flow [[buffer(0)]],
                                   constant uint2 &grid [[buffer(1)]],
                                   constant float2 &scale [[buffer(2)]],
                                   uint2 p [[thread_position_in_grid]]) {
   if (p.x>=output.get_width() || p.y>=output.get_height()) return;
   constexpr sampler linear(filter::linear,address::clamp_to_edge);
   float2 size=float2(output.get_width(),output.get_height());
   float2 uv=(float2(p)+.5f)/size;
   float2 point=clamp(uv*float2(grid)-.5f,float2(0.f),float2(grid-1));
   uint2 a=uint2(point), b=min(a+1,grid-1);
   float2 t=fract(point);
   float2 mv=mix(mix(flow[a.y*grid.x+a.x],flow[a.y*grid.x+b.x],t.x),
                 mix(flow[b.y*grid.x+a.x],flow[b.y*grid.x+b.x],t.x),t.y);
   float2 prior=uv+mv*scale/float2(grid);
   float3 value=float3(current.read(p).rgb);
   if (!all(isfinite(prior)) || any(prior<=0.f) || any(prior>=1.f)) {
    output.write(half4(half3(value),0),p);
    return;
   }
   float difference=length(appearance_decode(proxy.read(p).rgb)
                           -appearance_decode(previous.sample(linear,prior).rgb));
   // A center pixel can match across an occlusion. Require the neighboring
   // structure to agree before reusing the old correction.
   float confidence=min(1.f-smoothstep(.015f,.12f,difference),
                        patch_trust(previous,proxy,prior,uv,size));
   float3 low=0.f, oldLow=0.f;
   for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++) {
    float2 offset=float2(x,y)/size;
    low+=float3(current.sample(linear,uv+offset).rgb)/9.f;
    oldLow+=float3(history.sample(linear,prior+offset).rgb)/9.f;
   }
   float3 stable=value-low+mix(low,clamp(oldLow,low-.015f,low+.015f),.65f*confidence);
   output.write(half4(half3(stable),0),p);
  }
  // Low band of the signed correction, guided by the captured source.
  // Both base and intermediate frames use this same field in guided MFG.
  kernel void mfg_correction(texture2d<half,access::read> field [[texture(0)]],
                             texture2d<float,access::read> proxy [[texture(1)]],
                             texture2d<half,access::write> output [[texture(2)]],
                             uint2 p [[thread_position_in_grid]]) {
   if(p.x>=output.get_width() || p.y>=output.get_height()) return;
   int2 last=int2(output.get_width()-1,output.get_height()-1);
   float3 center=proxy.read(p).rgb;
   float3 low=0.f;
   float mass=0.f;
   for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++) {
    uint2 tap=uint2(clamp(int2(p)+int2(x,y),int2(0),last));
    float3 difference=proxy.read(tap).rgb-center;
    float weight=float((2-abs(x))*(2-abs(y)))*exp(-24.f*length(difference));
    low+=weight*float3(field.read(tap).rgb);
    mass+=weight;
   }
   output.write(half4(half3(low/mass),0),p);
  }
  kernel void compose_appearance(texture2d<float> field [[texture(0)]],
                                 texture2d<float,access::read> raw [[texture(1)]],
                                 texture2d<float,access::write> target [[texture(2)]],
                                 texture2d<float> proxy [[texture(3)]],
                                 constant float4 &hud [[buffer(0)]],
                                 constant float &gain [[buffer(1)]],
                                 constant float3 &colorControls [[buffer(2)]],
                                 uint2 p [[thread_position_in_grid]]) {
   if (p.x >= target.get_width() || p.y >= target.get_height()) return;
   float2 uv=(float2(p)+.5f)/float2(target.get_width(),target.get_height());
   float4 original=raw.read(p);
   if (gain == 0.f || (all(uv >= hud.xy) && all(uv < hud.zw))) {
    target.write(original,p);
    return;
   }
   CorrectionSample value=sample_correction(field,proxy,uv,original.rgb);
   float3 result=compose_source_detail(original.rgb,value.broad,value.delta,gain,colorControls);
   target.write(float4(result,1.f),p);
  }
  """,options:nil)
  unpack=try device.makeComputePipelineState(function:lib.makeFunction(name:"unpack")!)
  extractCorrection=try device.makeComputePipelineState(function:lib.makeFunction(name:"extract_correction")!)
  stabilizeCorrection=try device.makeComputePipelineState(function:lib.makeFunction(name:"stabilize_correction")!)
  mfgCorrection=try device.makeComputePipelineState(function:lib.makeFunction(name:"mfg_correction")!)
  composeAppearance=try device.makeComputePipelineState(function:lib.makeFunction(name:"compose_appearance")!)
  featuresKernel=MLXFast.metalKernel(name:"three_mode_features",inputNames:["base","rgb","history"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;if(p>=NW*NH)return;
  uint x=p%NW,y=p/NW,sx=x<W ? x : 2*W-2-x,sy=y<H ? y : 2*H-2-y;
  for(uint c=0;c<16;c++)out[p*16+c]=base[p*16+c];
  for(uint c=0;c<3;c++) {half s=half(rgb[(sy*W+sx)*3+c]);half centered=half(s-half(.5f));half v=half(centered*half(.125f));out[p*16+4+c]=v;out[p*16+7+c]=Temporal ? half(history[(sy*W+sx)*7+c]) : v;}
  """)
  screenColor=MLXFast.metalKernel(name:"nr_screen_controls",inputNames:["raw","styled","controls"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;
  if(p>=W*H)return;
  float2 uv=(float2(p%W,p/W)+.5f)/float2(W,H);
  bool protect=controls[3]>0.f && all(uv>=float2(controls[4],controls[5])) && all(uv<float2(controls[6],controls[7]));
  float3 original=float3(raw[p*3],raw[p*3+1],raw[p*3+2]);
  float3 model=float3(styled[p*3],styled[p*3+1],styled[p*3+2]);
  float3 result=model;
  if(protect) {
   result=original;
  } else {
   float3 weights=float3(.2126f,.7152f,.0722f);
   float3 delta=model-original;
   float luma=dot(delta,weights);
   float targetLuma=clamp(dot(original,weights)+controls[0]*luma,0.f,1.f);
   float3 chroma=original-dot(original,weights)+controls[1]*(delta-luma);
   float3 room=select(float3(targetLuma),float3(1.f-targetLuma),chroma>0.f);
   float3 ratios=room/max(abs(chroma),1e-6f);
   result=clamp(targetLuma+chroma*min(1.f,min(ratios.r,min(ratios.g,ratios.b))),0.f,1.f);
   if(controls[2]!=1.f) {
    float3 linear=select(result/12.92f,pow((result+.055f)/1.055f,2.4f),result>.04045f);
    linear=linear*controls[2]/(1.f+(controls[2]-1.f)*linear);
    result=select(linear*12.92f,1.055f*pow(linear,1.f/2.4f)-.055f,linear>.0031308f);
   }
  }
  for(uint c=0;c<3;c++)out[p*3+c]=result[c];
  """)
  alignHistory=MLXFast.metalKernel(name:"nr_align_history",inputNames:["current","previous","appearance","flow","scales"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;
  if(p>=W*H)return;
  float2 position=float2(p%W,p/W);
  float2 grid=(position+.5f)*float2(FW,FH)/float2(W,H)-.5f;
  grid=clamp(grid,float2(0),float2(FW-1,FH-1));
  uint2 g=uint2(floor(grid)), n=min(g+1,uint2(FW-1,FH-1));
  float2 f=fract(grid);
  uint indices[4]={g.y*FW+g.x,g.y*FW+n.x,n.y*FW+g.x,n.y*FW+n.x};
  float2 vectors[4];
  for(uint i=0;i<4;i++) vectors[i]=float2(flow[indices[i]*2],flow[indices[i]*2+1]);
  float2 delta=mix(mix(vectors[0],vectors[1],f.x),mix(vectors[2],vectors[3],f.x),f.y);
  float2 at=position+delta*float2(scales[0],scales[1])*float2(W,H)/float2(FW,FH);
  bool valid=all(isfinite(at)) && all(at>=0.f) && all(at<=float2(W-1,H-1));
  at=valid ? at:position;
  uint2 a=uint2(floor(at)), b=min(a+1,uint2(W-1,H-1));
  float2 t=fract(at);
  uint pixels[4]={a.y*W+a.x,a.y*W+b.x,b.y*W+a.x,b.y*W+b.x};
  float weights[4]={(1-t.x)*(1-t.y),t.x*(1-t.y),(1-t.x)*t.y,t.x*t.y};
  float3 raw=0,history=0,correction=0,now;
  for(uint c=0;c<3;c++){
   now[c]=float(current[p*3+c]);
   for(uint j=0;j<4;j++){
    raw[c]+=weights[j]*float(previous[pixels[j]*3+c]);
    history[c]+=weights[j]*float(appearance[pixels[j]*6+c]);
    correction[c]+=weights[j]*float(appearance[pixels[j]*6+3+c]);
   }
  }
  float3 error=raw-now;
  float confidence=valid ? 1.f-smoothstep(.01f,.12f,length(error)):0.f;
  for(uint c=0;c<3;c++) {
   half value=half(confidence>0.f ? history[c]:now[c]);
   out[p*7+c]=float(half(half(value-half(.5f))*half(.125f)));
   out[p*7+4+c]=correction[c];
  }
  out[p*7+3]=confidence;
  """)
  pack=MLXFast.metalKernel(name:"three_mode_pack4k",inputNames:["rgb","bias","reference","intensity"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;if(p>=W*H*9)return;
  uint x=p%(W*3),y=p/(W*3),sub=(y%6)*6+x%6,i=((y/6)*(W/2)+x/6)*108+sub;
  uint outPixel=y*PW+x;
  for(uint c=0;c<3;c++) {
   uint channel=i+c*36;
   float value=float(rgb[channel]+bias[sub+c*36]);
   if(intensity[0]!=1.f) {
    float baseline=clamp(float(reference[channel]+bias[sub+c*36]),0.f,1.f);
    value=baseline+intensity[0]*(clamp(value,0.f,1.f)-baseline);
   }
   out[outPixel*4+2-c]=uchar(clamp(round(value*255.f),0.f,255.f));
  }
  out[outPixel*4+3]=255;
  """)
  packRGB=MLXFast.metalKernel(name:"nr_pack_rgb",inputNames:["rgb"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;if(p>=W*H)return;uint q=(p/W)*PW+p%W;
  for(uint c=0;c<3;c++)out[q*4+2-c]=uchar(clamp(round(float(rgb[p*3+c])*255.f),0.f,255.f));out[q*4+3]=255;
  """)
  pack720=MLXFast.metalKernel(name:"three_mode_pack720",inputNames:["rgb"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;if(p>=1280*720)return;
  for(uint c=0;c<3;c++)out[p*4+2-c]=uchar(clamp(round(float(rgb[p*3+c])*255.f),0.f,255.f));out[p*4+3]=255;
  """)
 }
 private func texture(_ pixels:CVPixelBuffer)throws->MTLTexture {
  guard let surface=CVPixelBufferGetIOSurface(pixels)?.takeUnretainedValue() else {throw NSError(domain:"ThreeMode",code:3)}
  let d=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.bgra8Unorm,width:CVPixelBufferGetWidth(pixels),height:CVPixelBufferGetHeight(pixels),mipmapped:false)
  d.storageMode = .shared;d.usage=[.shaderRead,.shaderWrite]
  guard let t=device.makeTexture(descriptor:d,iosurface:surface,plane:0) else {throw NSError(domain:"ThreeMode",code:4)};return t
 }
 private func finish(_ cb:MTLCommandBuffer)throws {cb.commit();cb.waitUntilCompleted();guard cb.status == .completed else {throw cb.error ?? NSError(domain:"ThreeMode",code:5)}}
 private func read(_ input:CVPixelBuffer)throws->MLXArray {
  let width=CVPixelBufferGetWidth(input),height=CVPixelBufferGetHeight(input)
  guard (width==1920 && height==1080) || (width==1280 && height==720),
   CVPixelBufferGetPixelFormatType(input)==kCVPixelFormatType_32BGRA,
   let cb=queue.makeCommandBuffer() else {throw NSError(domain:"ThreeMode",code:6)}
  let src=try texture(input)
  let needsScale=width != lw || height != lh
  if needsScale {scale.encode(commandBuffer:cb,sourceTexture:src,destinationTexture:workingTexture)}
  guard let e=cb.makeComputeCommandEncoder() else {throw NSError(domain:"ThreeMode",code:6)}
  e.setComputePipelineState(unpack);e.setTexture(needsScale ? workingTexture : src,index:0);e.setBuffer(rgbBuffer,offset:0,index:0)
  e.dispatchThreads(MTLSize(width:lw,height:lh,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1));e.endEncoding();try finish(cb)
  let storage=rgbBuffer
  return MLXArray(rawPointer:storage.contents(),[1,lh,lw,3],dtype:.float32){_ = storage}
 }
 private func features(_ rgb:MLXArray, history:MLXArray?=nil)->MLXArray {
  featuresKernel([template!,rgb,history ?? rgb],template:[("Temporal",history != nil),("W",lw),("H",lh),("NW",nw),("NH",nh)],grid:(nw*nh,1,1),threadGroup:(256,1,1),outputShapes:[[1,nh,nw,16]],outputDTypes:[.float16])[0]
 }
 private func pass(_ rgb:MLXArray,model:MLXFullFrameRenderer, history:MLXArray?=nil)throws->(MLXArray,Double,Double) {
  let start=ProcessInfo.processInfo.systemUptime
  let f=features(rgb,history:history);eval(f)
  let prep=(ProcessInfo.processInfo.systemUptime-start)*1000
  let began=ProcessInfo.processInfo.systemUptime
  let head=model(f);eval(head)
  let ms=(ProcessInfo.processInfo.systemUptime-began)*1000
  guard head.shape == [1,nh,nw,4] else {throw NSError(domain:"ThreeMode",code:12,userInfo:[NSLocalizedDescriptionKey:"Unexpected padded head shape"])}
  guard all(isFinite(head)).item(Bool.self) else {throw NSError(domain:"ThreeMode",code:7)}
  let layerStrength = controls.layerGain
  let predicted=clip(rgb+head[0...,0..<lh,0..<lw,0..<3].asType(.float32)*(0.25*layerStrength),min:0,max:1)
  // Apply gain to the current prediction only. History already has the pass gain.
  if let history {
   let oldCorrection = history[0...,0...,0...,4..<7]
   let currentCorrection = predicted-rgb
   let alpha = sigmoid(head[0...,0..<lh,0..<lw,3..<4].asType(.float32))
     * Float(Float16(0.73974609375)) * history[0...,0...,0...,3..<4]
   // Limit disagreement before blending signed corrections. Geometry always comes
   // from this pass's current input, including in the second pass of a cascade.
   let difference = oldCorrection-currentCorrection
   let agreement = clip(1-abs(difference)*16,min:0,max:1)
   let output = clip(predicted + alpha * agreement * difference,min:0,max:1)
   eval(output)
   return (output,ms,prep)
  }
  eval(predicted)
  return(predicted,ms,prep)
 }
 private func pixels(_ packed:MLXArray,width:Int,height:Int)throws->(CVPixelBuffer,MTLTexture) {
  eval(packed)
  var out:CVPixelBuffer?
  let attrs:[CFString:Any]=[kCVPixelBufferIOSurfacePropertiesKey:[:],kCVPixelBufferMetalCompatibilityKey:true]
  guard CVPixelBufferCreate(nil,width,height,kCVPixelFormatType_32BGRA,attrs as CFDictionary,&out)==kCVReturnSuccess,let out,
   let buffer=packed.asMTLBuffer(device:device,noCopy:true),let cb=queue.makeCommandBuffer(),let blit=cb.makeBlitCommandEncoder() else {throw NSError(domain:"ThreeMode",code:8)}
  let t=try texture(out)
  blit.copy(from:buffer,sourceOffset:0,sourceBytesPerRow:packed.shape[1]*4,sourceBytesPerImage:packed.shape[1]*height*4,sourceSize:MTLSize(width:width,height:height,depth:1),to:t,destinationSlice:0,destinationLevel:0,destinationOrigin:MTLOrigin(x:0,y:0,z:0));blit.endEncoding();try finish(cb)
  withExtendedLifetime((packed,buffer)){};return(out,t)
 }
 func process(_ input:CVPixelBuffer,_ sequence:UInt64,_ pts:Double,captureCorrection:Bool=false)throws->ProcessedFrame {
  var completed = false
  defer {
   if !completed {
    previousRaw = nil
    passHistory = []
    correctionHistory = nil
    if #available(macOS 15.4, *) { (motion as? AppleFrameInterpolator)?.resetHistory() }
   }
  }
  let began=ProcessInfo.processInfo.systemUptime
  let readStart=ProcessInfo.processInfo.systemUptime
  let source=try read(input);frames+=1
  var stages:[String:Double]=["captureDownsampleUnpackMS":(ProcessInfo.processInfo.systemUptime-readStart)*1000,
   "captureWidth":Double(CVPixelBufferGetWidth(input)), "captureHeight":Double(CVPixelBufferGetHeight(input))]
  if template==nil {
   let values=source.asArray(Float.self)
   let host=try HostTensor(descriptor:TensorDescriptor(name:"color",shape:[1,lh,lw,3],dataType:.float32,layout:.nhwc),bytes:values.withUnsafeBytes{Data($0)})
   let cpu=try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(from:host,noiseFrameIndex:UInt32(controls.noiseFrame),geometry:geometry,normalizedStyle:Float(controls.style)/128,localToneStrength:controls.tone,localStructureStrength:controls.structure,automaticMask:controls.automatic,controlMask:nil)
   template=MLXArray(cpu.bytes,[1,nh,nw,16],dtype:.float32).asType(.float16);eval(template!)
   let check=Double(max(abs(features(source)-template!)).item(Float.self));stages["liveFeatureMaxAbs"]=check
   guard check==0 else {throw NSError(domain:"ThreeMode",code:9)}
  }
  var aligned: [MLXArray] = []
  let nativeCapture=lowResolution && CVPixelBufferGetWidth(input)==1920 && CVPixelBufferGetHeight(input)==1080
  let temporalEnabled=controls.appliesNR && controls.temporalNR == true
  let outputTemporal=temporalEnabled && nativeCapture
  let temporal=temporalEnabled && !outputTemporal
  var outputFlow:(Int,Int,[Float])?
  var sharedFlow:NativeFlowPair?
  if !outputTemporal { correctionHistory=nil }
  if temporalEnabled, #available(macOS 15.4, *) {
   let motionStart = ProcessInfo.processInfo.systemUptime
   if motion == nil { motion = try AppleFrameInterpolator(device: device) }
   let estimator = motion as! AppleFrameInterpolator
   let reuse=controls.frameGenerationFactor>1 && controls.frameGenerationBackend=="apple"
   let estimated=try estimator.estimateBackward(input,time:pts,retainPair:reuse)
   sharedFlow=estimator.completedHistoryFlow
   if outputTemporal { outputFlow=estimated }
   if temporal, let (width,height,values)=estimated, let previousRaw {
    let field=MLXArray(values,[height,width,2])
    let scales=MLXArray([controls.motionScaleX ?? 1,controls.motionScaleY ?? 1])
    for appearance in passHistory {
     let tensor=alignHistory([source,previousRaw,appearance,field,scales],
       template:[("W",lw),("H",lh),("FW",width),("FH",height)],
       grid:(lw*lh,1,1),threadGroup:(256,1,1),outputShapes:[[1,lh,lw,7]],outputDTypes:[.float32])[0]
     eval(tensor)
     aligned.append(tensor)
    }
   }
   if temporal {
    let owned=source.asType(.float16)
    eval(owned)
    previousRaw=owned
   }
   stages["nrScreenMotionMS"]=(ProcessInfo.processInfo.systemUptime-motionStart)*1000
   stages["nrTemporalPasses"]=Double(aligned.count)
  }
  var nextHistory: [MLXArray] = []
  let comparison=compareEvery>0 && (frames==1 || frames%compareEvery==0)
  var outputs:[String:MLXArray]=[:]
  var selected:MLXArray
  if !controls.appliesNR {selected=source
  } else if mode=="71" || mode=="71x2-480" || mode=="71-480" {
   let a=try pass(source,model:model71!,history:aligned.first);selected=a.0;if temporal {nextHistory.append(concatenated([a.0,a.0-source],axis:3).asType(.float16))};stages["pass1MS"]=a.1;stages["features1MS"]=a.2
  } else {
   let a=try pass(source,model:model31!,history:aligned.first);selected=a.0;if temporal {nextHistory.append(concatenated([a.0,a.0-source],axis:3).asType(.float16))};outputs["31"]=a.0;stages["pass1MS"]=a.1;stages["features1MS"]=a.2
  }
   if controls.appliesNR && ["31x2","71x2-480","31then71-480","31x2-480"].contains(mode) {
    let first=selected
    if frames==1 {
     let values=first.asArray(Float.self)
     let host=try HostTensor(descriptor:TensorDescriptor(name:"color",shape:[1,lh,lw,3],dataType:.float32,layout:.nhwc),bytes:values.withUnsafeBytes{Data($0)})
     let cpu=try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(from:host,noiseFrameIndex:UInt32(controls.noiseFrame),geometry:geometry,normalizedStyle:Float(controls.style)/128,localToneStrength:controls.tone,localStructureStrength:controls.structure,automaticMask:controls.automatic,controlMask:nil)
     let expected=MLXArray(cpu.bytes,[1,nh,nw,16],dtype:.float32).asType(.float16)
     let error=Double(max(abs(features(first)-expected)).item(Float.self));stages["pass2LiveFeatureMaxAbs"]=error
     guard error==0 else {throw NSError(domain:"ThreeMode",code:10)}
    }
    let b=try pass(first,model:(mode=="71x2-480" || mode=="31then71-480") ? model71! : model31!,history:aligned.count>1 ? aligned[1]:nil);selected=b.0;if temporal {nextHistory.append(concatenated([b.0,b.0-first],axis:3).asType(.float16))};stages["pass2MS"]=b.1;stages["features2MS"]=b.2
   }
  for entry in nextHistory { eval(entry) }
  passHistory=nextHistory
  let luma=controls.hdrStrength ?? 1, chroma=controls.colorStrength ?? 1
  let white=controls.diffuseWhite ?? 80
  let perceptualOutput=nativeCapture
  if !perceptualOutput && controls.appliesNR &&
     (luma != 1 || chroma != 1 || white != 80 || controls.uiCorrection == true) {
   let x=controls.uiX ?? 4.9, y=controls.uiY ?? 79
   let parameters=MLXArray([luma,chroma,white/80,controls.uiCorrection == true ? 1:0,
       x/100,y/100,(x+(controls.uiWidth ?? 14.7))/100,(y+(controls.uiHeight ?? 16.2))/100])
   selected=screenColor([source,selected,parameters],template:[("W",lw),("H",lh)],
     grid:(lw*lh,1,1),threadGroup:(256,1,1),outputShapes:[source.shape],outputDTypes:[.float32])[0]
   eval(selected)
  }
  outputs[mode]=selected
  let srStart=ProcessInfo.processInfo.systemUptime
  var output:(CVPixelBuffer,MTLTexture)
  var appearanceProxy: AppearanceProxy?
  var correctionEvidence: CorrectionEvidence?
  if nativeCapture {
   let pw=((lw+63)/64)*64
   let packed=packRGB([selected],template:[("W",lw),("H",lh),("PW",pw)],grid:(lw*lh,1,1),threadGroup:(256,1,1),outputShapes:[[lh,pw,4]],outputDTypes:[.uint8])[0]
   output=try pixels(packed,width:lw,height:lh)
  } else if controls.upscaling {
  let head=sr.head(selected.asType(.float16))
  let pw=((lw*3+63)/64)*64
  // Both branches use the same SR model and pixel grid. Apply the
  // final gain at packing, after the cascade and SR have completed.
  let strength:Float=controls.appliesNR ? controls.finalGain:1
  let reference=strength==1 ? head:sr.head(source.asType(.float16))
  let packed=pack([head,sr.outputBias,reference,MLXArray([strength])],template:[("W",lw),("H",lh),("PW",pw)],grid:(lw*lh*9,1,1),threadGroup:(256,1,1),outputShapes:[[lh*3,pw,4]],outputDTypes:[.uint8])[0]
  output=try pixels(packed,width:lw*3,height:lh*3)
  } else {
   let pw=((lw+63)/64)*64
   let rgb=controls.appliesNR && controls.finalGain != 1
    ? clip(source+(selected-source)*controls.finalGain,min:0,max:1):selected
   let packed=packRGB([rgb],template:[("W",lw),("H",lh),("PW",pw)],grid:(lw*lh,1,1),threadGroup:(256,1,1),outputShapes:[[lh,pw,4]],outputDTypes:[.uint8])[0]
   output=try pixels(packed,width:lw,height:lh)
  }
  if lowResolution {
   let resolveStart=ProcessInfo.processInfo.systemUptime
   var resolved:CVPixelBuffer?
   let attrs:[CFString:Any]=[kCVPixelBufferIOSurfacePropertiesKey:[:],kCVPixelBufferMetalCompatibilityKey:true]
   guard CVPixelBufferCreate(nil,1920,1080,kCVPixelFormatType_32BGRA,attrs as CFDictionary,&resolved)==kCVReturnSuccess,let resolved,let cb=queue.makeCommandBuffer() else {throw NSError(domain:"ThreeMode",code:11)}
   let target=try texture(resolved)
   if nativeCapture {
    let pw=((lw+63)/64)*64
    let packed=packRGB([source],template:[("W",lw),("H",lh),("PW",pw)],grid:(lw*lh,1,1),threadGroup:(256,1,1),outputShapes:[[lh,pw,4]],outputDTypes:[.uint8])[0]
    let proxy=try pixels(packed,width:lw,height:lh)
    let raw=try texture(input)
    if correctionTexture == nil {
     let descriptor=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba16Float,width:lw,height:lh,mipmapped:false)
     descriptor.storageMode = .private
     descriptor.usage = [.shaderRead,.shaderWrite]
     correctionTexture=device.makeTexture(descriptor:descriptor)
    }
    guard let field=correctionTexture else {throw NSError(domain:"ThreeMode",code:11)}
    func encodeCorrection(to target:MTLTexture, _ settings:SIMD3<Float>) throws {
     guard let encoder=cb.makeComputeCommandEncoder() else {throw NSError(domain:"ThreeMode",code:11)}
     var settings=settings
     encoder.setComputePipelineState(extractCorrection)
     encoder.setTexture(output.1,index:0);encoder.setTexture(proxy.1,index:1);encoder.setTexture(target,index:2)
     encoder.setBytes(&settings,length:MemoryLayout<SIMD3<Float>>.stride,index:0)
     encoder.dispatchThreads(MTLSize(width:lw,height:lh,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1))
     encoder.endEncoding()
    }
    let correctionControls:SIMD3<Float>=perceptualOutput ? SIMD3(luma,chroma,white/80) : SIMD3(repeating:1)
    try encodeCorrection(to:field,outputTemporal ? SIMD3(repeating:1) : correctionControls)
    var modelField:MTLTexture?
    if captureCorrection {
     let descriptor=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba16Float,width:lw,height:lh,mipmapped:false)
     descriptor.storageMode = .private
     descriptor.usage = [.shaderRead,.shaderWrite]
     if let texture=device.makeTexture(descriptor:descriptor) {
      try encodeCorrection(to:texture,outputTemporal ? correctionControls : SIMD3(repeating:1))
      modelField=texture
     }
    }
    var resolvedField=field
    if outputTemporal, let prior=correctionHistory, let (width,height,values)=outputFlow,
       pts>prior.pts, pts-prior.pts<=0.25 {
     if stableTexture == nil {
      let descriptor=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba16Float,width:lw,height:lh,mipmapped:false)
      descriptor.storageMode = .private
      descriptor.usage = [.shaderRead,.shaderWrite]
      stableTexture=device.makeTexture(descriptor:descriptor)
     }
     let motionBuffer=values.withUnsafeBytes { bytes in
      device.makeBuffer(bytes:bytes.baseAddress!,length:bytes.count,options:.storageModeShared)
     }
     guard let stable=stableTexture,let motionBuffer,let encoder=cb.makeComputeCommandEncoder() else {throw NSError(domain:"ThreeMode",code:11)}
     var grid=SIMD2<UInt32>(UInt32(width),UInt32(height))
     var motionScale=SIMD2<Float>(controls.motionScaleX ?? 1,controls.motionScaleY ?? 1)
     encoder.setComputePipelineState(stabilizeCorrection)
     encoder.setTexture(field,index:0);encoder.setTexture(prior.field,index:1)
     encoder.setTexture(proxy.1,index:2);encoder.setTexture(prior.proxy.1,index:3);encoder.setTexture(stable,index:4)
     encoder.setBuffer(motionBuffer,offset:0,index:0)
     encoder.setBytes(&grid,length:MemoryLayout<SIMD2<UInt32>>.stride,index:1)
     encoder.setBytes(&motionScale,length:MemoryLayout<SIMD2<Float>>.stride,index:2)
     encoder.dispatchThreads(MTLSize(width:lw,height:lh,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1))
     encoder.endEncoding()
     resolvedField=stable
     stages["outputLowFrequencyHistory"]=1
    }
    let postColor=outputTemporal ? correctionControls : SIMD3<Float>(repeating:1)
    let gain:Float=controls.appliesNR ? controls.finalGain:0
    let x=controls.uiX ?? 4.9, y=controls.uiY ?? 79
    let hud=controls.uiCorrection == true
     ? SIMD4<Float>(x,y,x+(controls.uiWidth ?? 14.7),y+(controls.uiHeight ?? 16.2))/100
     : SIMD4<Float>(repeating:0)
    var displayField=resolvedField
    if controls.captureGuided && controls.frameGenerationFactor > 1 && gain>0 {
     let descriptor=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba16Float,width:lw,height:lh,mipmapped:false)
     descriptor.storageMode = .private
     descriptor.usage = [.shaderRead,.shaderWrite]
     guard let snapshot=device.makeTexture(descriptor:descriptor),let filter=cb.makeComputeCommandEncoder() else {
      throw NSError(domain:"ThreeMode",code:11)
     }
     filter.setComputePipelineState(mfgCorrection)
     filter.setTexture(resolvedField,index:0);filter.setTexture(proxy.1,index:1);filter.setTexture(snapshot,index:2)
     filter.dispatchThreads(MTLSize(width:lw,height:lh,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1))
     filter.endEncoding()
     displayField=snapshot
     appearanceProxy=AppearanceProxy(source:proxy,correction:snapshot,colorControls:postColor,gain:gain,hud:hud)
     stages["mfgSharedLowBand"]=1
    } else if controls.captureGuided && controls.frameGenerationFactor > 1 {
     // Disabled NR still needs an explicit zero-gain endpoint.
     appearanceProxy=AppearanceProxy(source:proxy,correction:resolvedField,colorControls:postColor,gain:gain,hud:hud)
    }
    guard let encoder=cb.makeComputeCommandEncoder() else {throw NSError(domain:"ThreeMode",code:11)}
    encoder.setComputePipelineState(composeAppearance)
    encoder.setTexture(displayField,index:0);encoder.setTexture(raw,index:1);encoder.setTexture(target,index:2)
    encoder.setTexture(proxy.1,index:3)
    var gpuGain=gain, gpuColor=postColor, gpuHUD=hud
    encoder.setBytes(&gpuGain,length:MemoryLayout<Float>.stride,index:1)
    encoder.setBytes(&gpuColor,length:MemoryLayout<SIMD3<Float>>.stride,index:2)
    encoder.setBytes(&gpuHUD,length:MemoryLayout<SIMD4<Float>>.stride,index:0)
    encoder.dispatchThreads(MTLSize(width:1920,height:1080,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1))
    encoder.endEncoding()
    let rowBytes=((lw*8+255)/256)*256
    let byteCount=rowBytes*lh
    var readback:MTLBuffer?
    if let modelField,
       let buffer=device.makeBuffer(length:byteCount*3,options:.storageModeShared),
       let blit=cb.makeBlitCommandEncoder() {
     for (index,texture) in [modelField,field,resolvedField].enumerated() {
      blit.copy(from:texture,sourceSlice:0,sourceLevel:0,sourceOrigin:MTLOrigin(x:0,y:0,z:0),
                sourceSize:MTLSize(width:lw,height:lh,depth:1),to:buffer,
                destinationOffset:index*byteCount,destinationBytesPerRow:rowBytes,
                destinationBytesPerImage:byteCount)
     }
     blit.endEncoding()
     readback=buffer
    }
    try finish(cb)
    if let readback {
     correctionEvidence=CorrectionEvidence(model:Data(bytes:readback.contents().advanced(by:outputTemporal ? byteCount:0),count:byteCount),
       before:Data(bytes:readback.contents().advanced(by:outputTemporal ? 0:byteCount),count:byteCount),
       after:Data(bytes:readback.contents().advanced(by:byteCount*2),count:byteCount),
       width:lw,height:lh,rowBytes:rowBytes)
    }
    if outputTemporal {
     // Keep the consumed history alive through GPU completion, then rotate surfaces.
     if resolvedField !== field { stableTexture=correctionHistory?.field }
     else if correctionHistory?.field !== field { stableTexture=correctionHistory?.field }
     correctionHistory=(resolvedField,proxy,pts)
     if resolvedField === field { correctionTexture=nil }
    }
    stages["nativeCorrectionResolve"]=1
    stages["signedCorrectionFP16"]=1
    stages["perceptualCorrection"]=perceptualOutput ? 1:0
    stages["colorAfterHistory"]=outputTemporal ? 1:0
    stages["sourceLuminanceDetail"]=1
    withExtendedLifetime(proxy){}
   } else {
    scale.encode(commandBuffer:cb,sourceTexture:output.1,destinationTexture:target)
    try finish(cb)
   }
   withExtendedLifetime(output){}
   output=(resolved,target)
   stages["resolve1080MS"]=(ProcessInfo.processInfo.systemUptime-resolveStart)*1000
   stages["logicalWidth"]=Double(lw);stages["logicalHeight"]=Double(lh);stages["networkWidth"]=Double(nw);stages["networkHeight"]=Double(nh)
  }
  stages["srPackTransferMS"]=(ProcessInfo.processInfo.systemUptime-srStart)*1000
  var variants:[String:CVPixelBuffer]=[:]
  if comparison {
   let comparisonStart=ProcessInfo.processInfo.systemUptime
   if outputs["71"]==nil {outputs["71"]=try pass(source,model:model71!).0}
   if outputs["31"]==nil {outputs["31"]=try pass(source,model:model31!).0}
   if outputs["31x2"]==nil {outputs["31x2"]=try pass(outputs["31"]!,model:model31!).0}
   for name in ["31","31x2"] {
    let d=outputs[name]!-outputs["71"]!
    stages["\(name)RGBMAETo71"]=Double(mean(abs(d)).item(Float.self))
    let mse=Double(mean(d*d).item(Float.self));stages["\(name)PSNRTo71"]=mse>0 ? -10*log10(mse) : 120
   }
   stages["doubleVsSingle31MAE"]=Double(mean(abs(outputs["31x2"]!-outputs["31"]!)).item(Float.self))
   for name in ["71","31","31x2"] {
    let rgb=outputs[name]!
    let dx=rgb[0...,0...,1..<1280,0...]-rgb[0...,0...,0..<1279,0...]
    let dy=rgb[0...,1..<720,0...,0...]-rgb[0...,0..<719,0...,0...]
    stages["\(name)EdgeEnergy"]=Double((mean(abs(dx))+mean(abs(dy))).item(Float.self))/2
    stages["\(name)ClippedChannelFraction"]=Double(mean(((rgb .<= 0) .|| (rgb .>= 1)).asType(.float32)).item(Float.self))
    let p=pack720([rgb],grid:(1280*720,1,1),threadGroup:(256,1,1),outputShapes:[[720,1280,4]],outputDTypes:[.uint8])[0]
    variants[name]=try pixels(p,width:1280,height:720).0
   }
   stages["comparisonWallMS"]=(ProcessInfo.processInfo.systemUptime-comparisonStart)*1000
  }
  let frame=ProcessedFrame(output.0,output.1,sequence,pts,(ProcessInfo.processInfo.systemUptime-began)*1000,0)
  frame.appearanceProxy=appearanceProxy
  frame.nativeFlow=sharedFlow
  frame.correctionEvidence=correctionEvidence
  frame.sourcePixels=input;frame.stages=stages;frame.validation=comparison;frame.variants=variants
  completed = true
  return frame
 }
}
