import Foundation
import Metal

@main struct WarpRegression {
 static func main() throws {
  let d=MTLCreateSystemDefaultDevice()!,queue=d.makeCommandQueue()!
  let source=try String(contentsOfFile:CommandLine.arguments[1],encoding:.utf8)
  let lib=try d.makeLibrary(source:source,options:nil)
  let warp=try d.makeComputePipelineState(function:lib.makeFunction(name:"trajectory_pack")!)
  let w=1920,h=1080,cw=320,ch=180
  func texture(_ format:MTLPixelFormat)->MTLTexture {
   let td=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:format,width:w,height:h,mipmapped:false)
   td.storageMode = .shared;td.usage=[.shaderRead,.shaderWrite];return d.makeTexture(descriptor:td)!
  }
  let a=texture(.rgba32Float),b=texture(.rgba32Float),out=texture(.bgra8Unorm)
  let rawA=texture(.rgba32Float),rawB=texture(.rgba32Float),actual=texture(.rgba32Float)
  func image(shift:Float)->[Float] {
   var v=[Float](repeating:0,count:w*h*4)
   for y in 0..<h {for x in 0..<w {let i=(y*w+x)*4;v[i]=min(1,max(0,(Float(x)-shift)/Float(w)));v[i+1]=Float(y)/Float(h);v[i+2]=0.2;v[i+3]=1}}
   return v
  }
  func upload(_ image:[Float],_ t:MTLTexture) {image.withUnsafeBytes{t.replace(region:MTLRegionMake2D(0,0,w,h),mipmapLevel:0,withBytes:$0.baseAddress!,bytesPerRow:w*16)}}
  let original=image(shift:0);upload(original,a);upload(image(shift:24),b)
  func run(_ co:[Float],time:Float,guided:Bool=false)->[UInt8] {
   let field=co.withUnsafeBytes{d.makeBuffer(bytes:$0.baseAddress!,length:$0.count,options:.storageModeShared)!}
   var p:[UInt32]=[UInt32(ch),UInt32(cw),0,Float(6).bitPattern,time.bitPattern,guided ? 1:0]
   let cb=queue.makeCommandBuffer()!
   for state in [warp] {
    let e=cb.makeComputeCommandEncoder()!;e.setComputePipelineState(state)
    e.setTexture(a,index:0);e.setTexture(b,index:1);e.setTexture(out,index:2);e.setTexture(guided ? rawA:a,index:3);e.setTexture(guided ? rawB:b,index:4);e.setTexture(guided ? actual:a,index:5);e.setBuffer(field,offset:0,index:0)
    p.withUnsafeMutableBytes{e.setBytes($0.baseAddress!,length:$0.count,index:1)}
    e.dispatchThreads(MTLSize(width:w,height:h,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1));e.endEncoding()
   }
   cb.commit();cb.waitUntilCompleted();precondition(cb.status == .completed,"GPU dispatch failed")
   var bytes=[UInt8](repeating:0,count:w*h*4);bytes.withUnsafeMutableBytes{out.getBytes($0.baseAddress!,bytesPerRow:w*4,from:MTLRegionMake2D(0,0,w,h),mipmapLevel:0)}
   return bytes
  }
  var co=[Float](repeating:0,count:cw*ch*8)
  for i in 0..<(cw*ch) {co[i*8] = -2;co[i*8+2]=2}
  for k in 1...5 {
   let t=Float(k)/6,bytes=run(co,time:t)
   for y in stride(from:20,to:h-20,by:31) {for x in stride(from:40,to:w-40,by:29) {
    let expected=Int(round((Float(x)-24*t)/Float(w)*255)),actual=Int(bytes[(y*w+x)*4+2]);precondition(abs(expected-actual)<=1,"Known translation failed phase \(k) at \(x),\(y): \(actual) vs \(expected)")
   }}
  }
  // Occlusion/appearance disagreement must not crossfade red into blue.
  var red=[Float](repeating:0,count:w*h*4),blue=red
  for i in 0..<(w*h) {red[i*4]=1;red[i*4+3]=1;blue[i*4+2]=1;blue[i*4+3]=1}
  upload(red,a);upload(blue,b)
  co=[Float](repeating:0,count:cw*ch*8)
  for k in [1,5] {
   let bytes=run(co,time:Float(k)/6)
   for i in stride(from:0,to:w*h,by:193) {precondition(bytes[i*4] == (k<3 ? 0:255) && bytes[i*4+2] == (k<3 ? 255:0),"Rejected correspondence crossfaded endpoint colors")}
  }
  // Guided output must use one NR texture for every phase, not a crossfade
  // of independently generated neural colors or detail.
  var gray=[Float](repeating:0.2,count:w*h*4)
  for i in 0..<(w*h){gray[i*4+3]=1}
  upload(gray,rawA);upload(gray,rawB);upload(gray,actual)
  for k in 1...5 {
   let bytes=run(co,time:Float(k)/6,guided:true)
   for i in stride(from:0,to:w*h,by:193){precondition(bytes[i*4]==255 && bytes[i*4+2]==0,"Two NR anchors crossfaded in guided phase \(k)")}
  }
  // A new real object has a different raw color from either endpoint. No
  // endpoint blue or uniform grade may contaminate it.
  var real=gray
  for y in 250..<750 {for x in 700..<1100 {let i=(y*w+x)*4;real[i]=0.7;real[i+1]=0.15;real[i+2]=0.3}}
  upload(real,actual)
  let bytes=run(co,time:0.5,guided:true)
  for y in stride(from:270,to:730,by:19){for x in stride(from:720,to:1080,by:17){for c in 0..<3 {
   let expected=Int(round(real[(y*w+x)*4+c]*255));precondition(abs(Int(bytes[(y*w+x)*4+2-c])-expected)<=1,"Disoccluded capture contaminated by NR color")
  }}}
  // A folding/non-convergent trajectory must fade NR appearance, never fold
  // actual captured geometry or introduce endpoint color.
  for y in 0..<ch{for x in 0..<cw{let i=(y*cw+x)*8;co[i] = -2*Float(x);co[i+2] = -2*Float(x)}}
  upload(gray,actual);let folded=run(co,time:0.5,guided:true)
  for y in stride(from:200,to:800,by:37){for x in stride(from:500,to:1400,by:41){for c in 0..<3{precondition(abs(Int(folded[(y*w+x)*4+c])-51)<=1,"Folding trajectory transferred false appearance")}}}
  print("PASS: five-phase translation; single NR anchor at every phase; no RGB crossfade; rejected disoccluded color and folding trajectories")
 }
}
