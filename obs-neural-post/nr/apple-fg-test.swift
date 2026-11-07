import Foundation
import CoreVideo
import Metal

final class ProcessedFrame {
 let pixels:CVPixelBuffer,texture:MTLTexture,sequence:UInt64,sourceSeconds:Double
 var motionGuide:CaptureGuide?
 var captureGuided=false
 var generated=false,phase:Double=1,pairStart:UInt64=0,pairEnd:UInt64=0
 init(_ p:CVPixelBuffer,_ t:MTLTexture,_ s:UInt64,_ time:Double,_ ms:Double,_ gpu:Double){pixels=p;texture=t;sequence=s;sourceSeconds=time}
}
enum MultiFrameInterpolator {struct Result {let frames:[ProcessedFrame],milliseconds:Double,cut:Bool,difference:Double,validationMaxDelta:Int}}
@main struct NativeInterpolationTest {
 static func main()throws {
  guard #available(macOS 15.4,*) else{return}
  let d=MTLCreateSystemDefaultDevice()!
  func frame(_ seq:UInt64,_ time:Double)throws->ProcessedFrame {
   var p:CVPixelBuffer?;precondition(CVPixelBufferCreate(nil,1920,1080,kCVPixelFormatType_32BGRA,[kCVPixelBufferIOSurfacePropertiesKey:[:],kCVPixelBufferMetalCompatibilityKey:true] as CFDictionary,&p)==kCVReturnSuccess)
   let io=CVPixelBufferGetIOSurface(p!)!.takeUnretainedValue(),td=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.bgra8Unorm,width:1920,height:1080,mipmapped:false)
   td.storageMode = .shared;td.usage=[.shaderRead,.shaderWrite];let t=d.makeTexture(descriptor:td,iosurface:io,plane:0)!
   CVPixelBufferLockBaseAddress(p!,[]);let base=CVPixelBufferGetBaseAddress(p!)!.assumingMemoryBound(to:UInt8.self),stride=CVPixelBufferGetBytesPerRow(p!)
   for y in 0..<1080 {for x in 0..<1920 {let i=y*stride+x*4;base[i]=167;base[i+1]=93;base[i+2]=31;base[i+3]=255}}
   CVPixelBufferUnlockBaseAddress(p!,[])
   return ProcessedFrame(p!,t,seq,time,0,0)
  }
  let engine=try AppleFrameInterpolator(device:d)
  try engine.prepare()
  _ = try engine.generate(frame(1,1),factor:6)
  let result=try engine.generate(frame(2,1.1),factor:6)
  precondition(result.frames.count==5,"Native processor did not return five frames")
  var maximum=0
  for (i,f) in result.frames.enumerated() {
   precondition(f.generated && abs(f.phase-Double(Float(i+1)/6))<1e-6)
   precondition(abs(f.sourceSeconds-(1+0.1*f.phase))<1e-9 && f.pairStart==1 && f.pairEnd==2)
   CVPixelBufferLockBaseAddress(f.pixels,.readOnly)
   let p=CVPixelBufferGetBaseAddress(f.pixels)!.assumingMemoryBound(to:UInt8.self),stride=CVPixelBufferGetBytesPerRow(f.pixels)
   for y in Swift.stride(from:24,to:1050,by:31) {for x in Swift.stride(from:24,to:1900,by:29) {for (c,v) in [167,93,31,255].enumerated(){maximum=max(maximum,abs(Int(p[y*stride+x*4+c])-v))}}}
   CVPixelBufferUnlockBaseAddress(f.pixels,.readOnly)
  }
  precondition(maximum<=2,"Color conversion/channel parity failed: \(maximum)")
  engine.reset()
  let ga=try frame(10,2),gb=try frame(16,2.1)
  ga.motionGuide=CaptureGuide(texture:ga.texture,sequence:10,seconds:2);gb.motionGuide=CaptureGuide(texture:gb.texture,sequence:16,seconds:2.1)
  var actuals:[ProcessedFrame]=[],guides:[CaptureGuide]=[]
  for i in 1...5 {
   let f=try frame(UInt64(10+i),2+Double(i)/60)
   CVPixelBufferLockBaseAddress(f.pixels,[]);let p=CVPixelBufferGetBaseAddress(f.pixels)!.assumingMemoryBound(to:UInt8.self),row=CVPixelBufferGetBytesPerRow(f.pixels)
   for y in 200..<500 {for x in (60+i*i*30)..<(160+i*i*30) {let j=y*row+x*4;p[j]=40;p[j+1]=170;p[j+2]=80}}
   CVPixelBufferUnlockBaseAddress(f.pixels,[])
   actuals.append(f);guides.append(CaptureGuide(texture:f.texture,sequence:UInt64(10+i),seconds:f.sourceSeconds))
  }
  _ = try engine.generate(ga,factor:6,guides:guides,guided:true)
  let guided=try engine.generate(gb,factor:6,guides:guides,guided:true)
  precondition(guided.frames.count==5,"Native guided output missing")
  var guideDelta=0
  for (i,f) in guided.frames.enumerated() {
   precondition(f.captureGuided && abs(f.sourceSeconds-guides[i].seconds)<1e-9)
   CVPixelBufferLockBaseAddress(f.pixels,.readOnly);CVPixelBufferLockBaseAddress(actuals[i].pixels,.readOnly)
   let p=CVPixelBufferGetBaseAddress(f.pixels)!.assumingMemoryBound(to:UInt8.self),r=CVPixelBufferGetBaseAddress(actuals[i].pixels)!.assumingMemoryBound(to:UInt8.self),ps=CVPixelBufferGetBytesPerRow(f.pixels),rs=CVPixelBufferGetBytesPerRow(actuals[i].pixels)
   for y in stride(from:20,to:1060,by:13) {for x in stride(from:20,to:1900,by:11) {for c in 0..<3 {guideDelta=max(guideDelta,abs(Int(p[y*ps+x*4+c])-Int(r[y*rs+x*4+c])))}}}
   CVPixelBufferUnlockBaseAddress(actuals[i].pixels,.readOnly);CVPixelBufferUnlockBaseAddress(f.pixels,.readOnly)
  }
  precondition(guideDelta<=2,"Native guided geometry changed despite zero appearance difference: \(guideDelta)")
  engine.reset()
  let sa=try frame(10,2),sb=try frame(16,2.1)
  sa.motionGuide=ga.motionGuide;sb.motionGuide=gb.motionGuide
  for f in [sa,sb] {
   CVPixelBufferLockBaseAddress(f.pixels,[])
   let p=CVPixelBufferGetBaseAddress(f.pixels)!.assumingMemoryBound(to:UInt8.self),row=CVPixelBufferGetBytesPerRow(f.pixels)
   for y in 0..<1080 {for x in 0..<1920 {for c in 0..<3 {p[y*row+x*4+c]+=20}}}
   CVPixelBufferUnlockBaseAddress(f.pixels,[])
  }
  _ = try engine.generate(sa,factor:6,guides:guides,guided:true)
  let tone=try engine.generate(sb,factor:6,guides:guides,guided:true)
  precondition(tone.frames.count==5)
  var toneDelta=0
  for (i,f) in tone.frames.enumerated() {
   CVPixelBufferLockBaseAddress(f.pixels,.readOnly);CVPixelBufferLockBaseAddress(actuals[i].pixels,.readOnly)
   let p=CVPixelBufferGetBaseAddress(f.pixels)!.assumingMemoryBound(to:UInt8.self),r=CVPixelBufferGetBaseAddress(actuals[i].pixels)!.assumingMemoryBound(to:UInt8.self),ps=CVPixelBufferGetBytesPerRow(f.pixels),rs=CVPixelBufferGetBytesPerRow(actuals[i].pixels)
   for y in stride(from:20,to:1060,by:13) {for x in stride(from:20,to:1900,by:11) {for c in 0..<3 {let lo=60+(i+1)*(i+1)*30,hi=lo+100
    if (abs(x-lo)<5 || abs(x-hi)<5) && y>195 && y<505 {continue}
    if (abs(y-200)<5 || abs(y-500)<5) && x>lo-5 && x<hi+5 {continue}
    let expected=(x>=lo && x<hi && y>=200 && y<500) ? 0:20
    toneDelta=max(toneDelta,abs(Int(p[y*ps+x*4+c])-Int(r[y*rs+x*4+c])-expected))}}}
   CVPixelBufferUnlockBaseAddress(actuals[i].pixels,.readOnly);CVPixelBufferUnlockBaseAddress(f.pixels,.readOnly)
  }
  precondition(toneDelta<=2,"Single-anchor visible appearance gain mismatch: \(toneDelta)")
  for f in [sa,sb] {
   CVPixelBufferLockBaseAddress(f.pixels,[]);let p=CVPixelBufferGetBaseAddress(f.pixels)!.assumingMemoryBound(to:UInt8.self),row=CVPixelBufferGetBytesPerRow(f.pixels)
   for y in 200..<500 {for x in 0..<1200 {p[y*row+x*4+1]+=60}}
   CVPixelBufferUnlockBaseAddress(f.pixels,[])
  }
  engine.reset();_ = try engine.generate(sa,factor:6,guides:guides,guided:true)
  let color=try engine.generate(sb,factor:6,guides:guides,guided:true)
  precondition(color.frames.count==5)
  for (i,f) in color.frames.enumerated() {
   CVPixelBufferLockBaseAddress(f.pixels,.readOnly);let p=CVPixelBufferGetBaseAddress(f.pixels)!.assumingMemoryBound(to:UInt8.self),row=CVPixelBufferGetBytesPerRow(f.pixels),x=110+(i+1)*(i+1)*30,j=350*row+x*4
   precondition(abs(Int(p[j+1])-170)<=2,"Unrelated native endpoint green contaminated intermediate object: \(p[j+1])")
   CVPixelBufferUnlockBaseAddress(f.pixels,.readOnly)
  }
  print("PASS: native phase/color; preserved visible single-anchor appearance; real intermediate geometry; rejected unrelated endpoint color")
 }
}
