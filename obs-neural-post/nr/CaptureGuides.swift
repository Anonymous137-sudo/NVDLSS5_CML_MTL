import Foundation
import Metal
import CoreVideo

fileprivate final class CaptureGuideSurface {let texture:MTLTexture;init(_ texture:MTLTexture){self.texture=texture}}
struct CaptureGuide {
 fileprivate let surface:CaptureGuideSurface
 var texture:MTLTexture {surface.texture}
 let sequence:UInt64,seconds:Double
 init(texture:MTLTexture,sequence:UInt64,seconds:Double) {surface=CaptureGuideSurface(texture);self.sequence=sequence;self.seconds=seconds}
 fileprivate init(surface:CaptureGuideSurface,sequence:UInt64,seconds:Double) {self.surface=surface;self.sequence=sequence;self.seconds=seconds}
}
/// Own copies of capture pixels; never retain an AVCapture pool buffer. CPU
/// upload avoids queuing extra GPU work or waiting on a busy GPU in capture.
final class CaptureGuideRing {
 private let device:MTLDevice,lock=NSLock()
 private var frames:[CaptureGuide]=[],copyTimes:[Double]=[]
 private var pool:[CaptureGuideSurface]=[]
 private var copied:UInt64=0,failed:UInt64=0,epoch:UInt64=0
 init(device:MTLDevice) {self.device=device}
 func copy(_ input:CVPixelBuffer,sequence:UInt64,seconds:Double,epoch captureEpoch:UInt64)->CaptureGuide? {
  let began=ProcessInfo.processInfo.systemUptime,w=CVPixelBufferGetWidth(input),h=CVPixelBufferGetHeight(input)
  lock.lock();guard captureEpoch==epoch else {lock.unlock();return nil};var lease:CaptureGuideSurface?
  for i in pool.indices where pool[i].texture.width==w && pool[i].texture.height==h {if isKnownUniquelyReferenced(&pool[i]) {lease=pool[i];break}}
  if lease==nil && pool.count<64 {
   let td=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.bgra8Unorm,width:w,height:h,mipmapped:false);td.storageMode = .shared;td.usage=[.shaderRead]
   if let t=device.makeTexture(descriptor:td) {let surface=CaptureGuideSurface(t);pool.append(surface);lease=surface}
  }
  lock.unlock()
  guard CVPixelBufferGetPixelFormatType(input)==kCVPixelFormatType_32BGRA,let lease,CVPixelBufferLockBaseAddress(input,.readOnly)==kCVReturnSuccess else {lock.lock();failed &+= 1;lock.unlock();return nil}
  let texture=lease.texture
  if let bytes=CVPixelBufferGetBaseAddress(input) {texture.replace(region:MTLRegionMake2D(0,0,w,h),mipmapLevel:0,withBytes:bytes,bytesPerRow:CVPixelBufferGetBytesPerRow(input))}
  else {CVPixelBufferUnlockBaseAddress(input,.readOnly);lock.lock();failed &+= 1;lock.unlock();return nil}
  CVPixelBufferUnlockBaseAddress(input,.readOnly)
  let frame=CaptureGuide(surface:lease,sequence:sequence,seconds:seconds)
  lock.lock();guard captureEpoch==epoch else {lock.unlock();return nil};frames.append(frame);if frames.count>32 {frames.removeFirst(frames.count-32)};copied &+= 1
  copyTimes.append((ProcessInfo.processInfo.systemUptime-began)*1000);if copyTimes.count>120 {copyTimes.removeFirst(copyTimes.count-120)};lock.unlock()
  return frame
 }
 func clear(epoch newEpoch:UInt64) {lock.lock();epoch=newEpoch;frames=[];lock.unlock()}
 func consecutivePath(from first:CaptureGuide,to last:CaptureGuide,epoch requestedEpoch:UInt64)->[CaptureGuide]? {
  lock.lock();defer{lock.unlock()}
  guard requestedEpoch==epoch,first.sequence<last.sequence,first.seconds<last.seconds else{return nil}
  let middle=frames.filter{$0.sequence>first.sequence && $0.sequence<last.sequence}
  let path=[first]+middle+[last]
  guard zip(path,path.dropFirst()).allSatisfy({a,b in
   b.sequence>a.sequence && b.sequence-a.sequence==1 && b.seconds>a.seconds && b.seconds-a.seconds<=0.05
  }) else{return nil}
  return path
 }
 func snapshot()->[CaptureGuide] {lock.lock();defer{lock.unlock()};return frames}
 func status()->[String:Any] {lock.lock();defer{lock.unlock()};let times=copyTimes.sorted();return ["copied":copied,"failed":failed,"retained":frames.count,"allocatedSurfaces":pool.count,"copyMedianMS":times.isEmpty ? 0:times[times.count/2],"copyMaxMS":times.last ?? 0]}
}

enum CaptureGuideSelection {
 static func select(_ frames:[CaptureGuide],after a:Double,before b:Double,factor:Int)->[CaptureGuide] {
  let candidates=frames.filter{$0.seconds>a+0.0001 && $0.seconds<b-0.0001}.sorted{$0.seconds<$1.seconds}
  guard candidates.count>factor-1 else{return candidates}
  var used=Set<UInt64>(),result:[CaptureGuide]=[]
  for i in 1..<factor {
   let target=a+(b-a)*Double(i)/Double(factor)
   if let best=candidates.filter({!used.contains($0.sequence)}).min(by:{abs($0.seconds-target)<abs($1.seconds-target)}) {used.insert(best.sequence);result.append(best)}
  }
  return result.sorted{$0.seconds<$1.seconds}
 }
}

/// Fit local exposure and chroma from an NR anchor, then apply them to the
/// real intermediate frame. No spatial frequencies from the anchor are copied.
let guidedAppearanceMetal = """
// Metal port of renodx::color::srgb Encode/Decode from src/shaders/color/srgb.hlsl.
float3 appearance_decode(float3 v) {
 return select(pow(fma(v,1.f/1.055f,.055f/1.055f),2.4f),v/12.92f,v<=.04045f);
}
float3 appearance_encode(float3 v) {
 v=clamp(v,0.f,1.f);
 return select(fma(1.055f,pow(v,1.f/2.4f),-.055f),v*12.92f,v<=.0031308f);
}
float3 compose_correction(float3 guide,float3 delta) {
 if (all(delta == 0.f)) return guide;
 float3 corrected=appearance_decode(guide)+delta;
 const float3 weights=float3(.2126f,.7152f,.0722f);
 float luma=clamp(dot(corrected,weights),0.f,1.f);
 float3 chroma=corrected-dot(corrected,weights);
 float3 room=select(float3(luma),float3(1.f-luma),chroma>0.f);
 float3 limit=room/max(abs(chroma),1e-6f);
 return appearance_encode(luma+chroma*min(1.f,min(limit.r,min(limit.g,limit.b))));
}
float3 apply_guided_appearance(float3 guide,float3 reference,float3 styled,float confidence) {
 return compose_correction(guide,confidence*(appearance_decode(styled)-appearance_decode(reference)));
}
  // BT.709/OKLab matrices from RenoDX src/shaders/color/oklab.hlsl.
  // Perceptual delta and constant-lightness gamut search follow the OBS
  // DLSS5 Neural Filter's controlled() path in data/neural.hlsl.
  float3 linear_to_oklab(float3 c) {
      float3 l = float3(dot(c,float3(.4122214708,.5363325363,.0514459929)),
          dot(c,float3(.2119034982,.6806995451,.1073969566)),
          dot(c,float3(.0883024619,.2817188376,.6299787005)));
      l = sign(l)*pow(abs(l),1.f/3.f);
      return float3(dot(l,float3(.2104542553,.793617785,-.0040720468)),
          dot(l,float3(1.9779984951,-2.428592205,.4505937099)),
          dot(l,float3(.0259040371,.7827717662,-.808675766)));
  }
  float3 oklab_to_linear(float3 c) {
      float3 l = float3(c.x+.3963377774*c.y+.2158037573*c.z,
          c.x-.1055613458*c.y-.0638541728*c.z,c.x-.0894841775*c.y-1.291485548*c.z);
      l = l*l*l;
      return float3(dot(l,float3(4.0767416621,-3.3077115913,.2309699292)),
          dot(l,float3(-1.2684380046,2.6097574011,-.3413193965)),
          dot(l,float3(-.0041960863,-.7034186147,1.707614701)));
  }
  float3 perceptual_correction(float3 source,float3 styled,float lightness,float chroma) {
   if (all(source == styled)) return float3(0.f);
   if (lightness == 1.f && chroma == 1.f) return styled-source;
   float3 base=linear_to_oklab(source);
   float3 difference=linear_to_oklab(styled)-base;
   float3 lab=base+difference*float3(lightness,chroma,chroma);
   lab.x=clamp(lab.x,0.f,1.f);
   float3 result=oklab_to_linear(lab);
   if (any(result<0.f) || any(result>1.f)) {
    float low=0.f, high=1.f;
    for(int i=0;i<16;i++) {
     float mid=(low+high)*.5f;
     float3 trial=oklab_to_linear(float3(lab.x,lab.yz*mid));
     if (all(trial>=0.f) && all(trial<=1.f)) low=mid;
     else high=mid;
    }
    result=oklab_to_linear(float3(lab.x,lab.yz*low));
   }
   return clamp(result,0.f,1.f)-source;
  }
float3 compose_source_detail(float3 original,float3 broad,float3 correction,float gain,float3 colorControls) {
 if (gain==0.f || all(correction==0.f)) return original;
   if (any(colorControls!=1.f)) {
    float3 source=appearance_decode(broad);
    float3 model=clamp(source+correction,0.f,1.f);
    if (colorControls.z!=1.f) model=model*colorControls.z/(1.f+(colorControls.z-1.f)*model);
    correction=perceptual_correction(source,model,colorControls.x,colorControls.y);
   }
   float3 styled=compose_correction(broad,gain*correction);
   const float3 lumaWeights=float3(.2126f,.7152f,.0722f);
   float broadLuma=dot(broad,lumaWeights), styledLuma=dot(styled,lumaWeights);
   float detail=dot(original,lumaWeights)-broadLuma;
   // Reserve room for the captured detail before placing the broad tone.
   float tone=clamp(styledLuma,max(0.f,-detail),min(1.f,1.f-detail));
   float luma=tone+detail;
   float3 chroma=original-dot(original,lumaWeights)
                 +(styled-styledLuma)-(broad-broadLuma);
   float3 room=select(float3(luma),float3(1.f-luma),chroma>0.f);
   float3 ratios=room/max(abs(chroma),1e-6f);
   return luma+chroma*min(1.f,min(ratios.r,min(ratios.g,ratios.b)));
}
float patch_trust(texture2d<float> endpoint, texture2d<float> actual, float2 uv, float2 here, float2 size) {
 constexpr sampler sample(filter::linear,address::clamp_to_edge);
 float2 border=2.5f/size;
 if (!all(isfinite(uv)) || any(uv<border) || any(uv>1.f-border)) return 0.f;
 float3 center=endpoint.sample(sample,uv).rgb-actual.sample(sample,here).rgb;
 float error=dot(center,center);
 float structure=0.f;
 for(int axis=0;axis<2;axis++) for(int sign=-1;sign<=1;sign+=2) {
  float2 offset=float(sign)*(axis==0 ? float2(2,0) : float2(0,2))/size;
  float3 mismatch=endpoint.sample(sample,uv+offset).rgb-actual.sample(sample,here+offset).rgb;
  error+=dot(mismatch,mismatch);
  float3 edge=mismatch-center;
  structure+=dot(edge,edge);
 }
 return exp(-error/.0125f-structure/.01f);
}
// Endpoint flow predicts a location at the intermediate timestamp. Refine
// that prediction against the actual captured intermediate along the motion
// axis. Keep the search small so it cannot jump to a different object.
float3 register_guide(texture2d<float> endpoint,texture2d<float> actual,
                      float2 prediction,float2 here,float2 size) {
 float trust=patch_trust(endpoint,actual,prediction,here,size);
 float2 best=prediction;
 float2 motion=(prediction-here)*size;
 float distance=length(motion);
 if (trust>=.90f || distance<1.f || !isfinite(distance)) return float3(best,trust);
 float2 step=motion/distance*min(2.f,distance*.25f)/size;
 for(int direction=-1;direction<=1;direction+=2) {
  float2 candidate=prediction+float(direction)*step;
  float score=patch_trust(endpoint,actual,candidate,here,size)*.95f;
  if (score>trust) {trust=score;best=candidate;}
 }
 return float3(best,trust);
}
struct CorrectionSample { float3 broad; float3 delta; };
CorrectionSample sample_correction(texture2d<float> field,texture2d<float> proxy,float2 uv,float3 original) {
 if (!all(isfinite(uv))) return {original,float3(0.f)};
 uv=clamp(uv,0.f,1.f);
   float2 position=uv*float2(field.get_width(),field.get_height())-.5f;
   int2 origin=int2(floor(position)), last=int2(field.get_width()-1,field.get_height()-1);
   float2 fraction=fract(position);
   float3 delta=0.f, broad=0.f;
   float mass=0.f;
   for (int y=0;y<2;y++) for (int x=0;x<2;x++) {
    uint2 tap=uint2(clamp(origin+int2(x,y),int2(0),last));
    float3 reference=proxy.read(tap).rgb;
    float3 difference=reference-original;
    float spatial=(x==0 ? 1.f-fraction.x : fraction.x)*(y==0 ? 1.f-fraction.y : fraction.y);
    float weight=spatial/(1.f+32.f*dot(difference,difference));
    delta+=weight*float3(field.read(tap).rgb);
    broad+=weight*reference;
    mass+=weight;
   }
 return {broad/mass,delta/mass};
}
float3 compose_transported_detail(float3 guide,CorrectionSample a,CorrectionSample b,float phase,
                                  float trustA,float trustB,float gain,float3 controls) {
 float wa=(1.f-phase)*trustA, wb=phase*trustB;
 float mass=wa+wb;
 if (!(mass>1e-5f)) return guide;
 // Confidence limits the effect. It must never darken the reference used
 // by the nonlinear color transform.
 float support=smoothstep(.08f,.55f,max(trustA,trustB));
 float3 broad=(wa*a.broad+wb*b.broad)/mass;
 float3 delta=(wa*a.delta+wb*b.delta)/mass;
 return compose_source_detail(guide,broad,delta,gain*support,controls);
}
"""
