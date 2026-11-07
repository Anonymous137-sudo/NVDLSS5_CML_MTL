import Foundation
import Metal
import CoreVideo
@main struct GuideTests {
 static func main() {
  let d=MTLCreateSystemDefaultDevice()!,td=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.bgra8Unorm,width:1,height:1,mipmapped:false);td.storageMode = .shared
  let texture=d.makeTexture(descriptor:td)!,guides=(0...12).map{CaptureGuide(texture:texture,sequence:UInt64($0),seconds:Double($0)/60)}
  precondition(CaptureGuideSelection.select(guides,after:0,before:0.1,factor:6).map(\.sequence)==[1,2,3,4,5])
  precondition(CaptureGuideSelection.select(guides,after:0,before:2.0/60,factor:6).map(\.sequence)==[1])
  precondition(CaptureGuideSelection.select(guides,after:0,before:0.2,factor:3).map(\.sequence)==[4,8])
  var p:CVPixelBuffer?;precondition(CVPixelBufferCreate(nil,1,1,kCVPixelFormatType_32BGRA,nil,&p)==kCVReturnSuccess)
  let ring=CaptureGuideRing(device:d);var first:CaptureGuide?
  for i in 0..<200 {
   CVPixelBufferLockBaseAddress(p!,[]);let b=CVPixelBufferGetBaseAddress(p!)!.assumingMemoryBound(to:UInt8.self);b[0]=UInt8(i%256);b[1]=13;b[2]=23;b[3]=255;CVPixelBufferUnlockBaseAddress(p!,[])
   let copy=ring.copy(p!,sequence:UInt64(i),seconds:Double(i)/60);precondition(copy != nil);if i==0 {first=copy}
  }
  var bytes=[UInt8](repeating:0,count:4);bytes.withUnsafeMutableBytes{first!.texture.getBytes($0.baseAddress!,bytesPerRow:4,from:MTLRegionMake2D(0,0,1,1),mipmapLevel:0)}
  precondition(bytes==[0,13,23,255],"Guide aliased the reused capture buffer")
  precondition(ring.snapshot().count==32 && (ring.status()["failed"] as? UInt64)==0)
  precondition((ring.status()["allocatedSurfaces"] as? Int ?? 999)<=36,"Guide upload pool failed to reuse retired owned surfaces")
  ring.clear();precondition(ring.snapshot().isEmpty)
  print("PASS: unique real capture phases; no invented phase duplication; bounded32-frame ring; immutable capture ownership; reset")
 }
}
