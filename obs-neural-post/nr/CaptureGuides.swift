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
 private var copied:UInt64=0,failed:UInt64=0
 init(device:MTLDevice) {self.device=device}
 func copy(_ input:CVPixelBuffer,sequence:UInt64,seconds:Double)->CaptureGuide? {
  let began=ProcessInfo.processInfo.systemUptime,w=CVPixelBufferGetWidth(input),h=CVPixelBufferGetHeight(input)
  lock.lock();var lease:CaptureGuideSurface?
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
  lock.lock();frames.append(frame);if frames.count>32 {frames.removeFirst(frames.count-32)};copied &+= 1
  copyTimes.append((ProcessInfo.processInfo.systemUptime-began)*1000);if copyTimes.count>120 {copyTimes.removeFirst(copyTimes.count-120)};lock.unlock()
  return frame
 }
 func clear() {lock.lock();frames=[];lock.unlock()}
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
float3 apply_guided_appearance(float3 guide, float3 reference, float3 styled, float confidence) {
 const float3 weights=float3(.2126f,.7152f,.0722f);
 float rawLuma=dot(reference,weights),styledLuma=dot(styled,weights);
 float gain=(styledLuma+.04f)/(rawLuma+.04f);
 float guideLuma=dot(guide,weights);
 float luma=mix(guideLuma,clamp((guideLuma+.04f)*gain-.04f,0.f,1.f),confidence);
 float3 chroma=guide-guideLuma;
 chroma+=confidence*((styled-styledLuma)-(reference-rawLuma));
 // Keep the target luminance and hue when the color reaches the RGB gamut.
 float3 room=select(float3(luma),float3(1.f-luma),chroma>0.f);
 float3 limit=room/max(abs(chroma),1e-6f);
 float chromaScale=min(1.f,min(limit.r,min(limit.g,limit.b)));
 return luma+chroma*chromaScale;
}
"""
