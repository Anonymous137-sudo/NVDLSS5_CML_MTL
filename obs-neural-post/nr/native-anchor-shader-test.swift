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
@main struct NativeAnchorShaderTest {
 static func main()throws {
  guard #available(macOS 15.4,*) else{return}
  let d=MTLCreateSystemDefaultDevice()!,queue=d.makeCommandQueue()!,lib=try d.makeLibrary(source:AppleFrameInterpolator.shader,options:nil),state=try d.makeComputePipelineState(function:lib.makeFunction(name:"anchor_pack")!)
  let w=1920,h=1080,fw=240,fh=135
  func texture(_ f:MTLPixelFormat,_ x:Int,_ y:Int)->MTLTexture{let t=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:f,width:x,height:y,mipmapped:false);t.storageMode = .shared;t.usage=[.shaderRead,.shaderWrite];return d.makeTexture(descriptor:t)!}
  let raw=texture(.rgba16Float,w,h),style=texture(.rgba16Float,w,h),actual=texture(.rgba16Float,w,h),out=texture(.bgra8Unorm,w,h),forward=texture(.rg16Float,fw,fh),backward=texture(.rg16Float,fw,fh)
  func image(_ shiftX:Float,_ shiftY:Float,_ tone:Float)->[Float16]{var p=[Float16](repeating:1,count:w*h*4);for y in 0..<h{for x in 0..<w{let i=(y*w+x)*4;p[i]=Float16(min(1,max(0,(Float(x)-shiftX)/Float(w)+tone)));p[i+1]=Float16(min(1,max(0,(Float(y)-shiftY)/Float(h)+tone)));p[i+2]=Float16(0.2+tone)}};return p}
  func upload(_ a:[Float16],_ t:MTLTexture,_ channels:Int=4){a.withUnsafeBytes{t.replace(region:MTLRegionMake2D(0,0,t.width,t.height),mipmapLevel:0,withBytes:$0.baseAddress!,bytesPerRow:t.width*channels*2)}}
  upload(image(32,-24,0),raw);upload(image(32,-24,0.08),style)
  var f=[Float16](repeating:0,count:fw*fh*2),b=f
  for i in 0..<(fw*fh){f[i*2]=4;f[i*2+1] = -3;b[i*2] = -4;b[i*2+1]=3};upload(f,forward,2);upload(b,backward,2)
  var maximum=0
  for k in 1...5 {
   let t=Float(k)/6;upload(image(32*t,-24*t,0),actual)
   let cb=queue.makeCommandBuffer()!,e=cb.makeComputeCommandEncoder()!;e.setComputePipelineState(state)
   for (i,t) in [style,out,raw,actual,backward,forward].enumerated(){e.setTexture(t,index:i)}
   var params:[UInt32]=[t.bitPattern,1];params.withUnsafeMutableBytes{e.setBytes($0.baseAddress!,length:$0.count,index:0)}
   e.dispatchThreads(MTLSize(width:w,height:h,depth:1),threadsPerThreadgroup:MTLSize(width:16,height:8,depth:1));e.endEncoding();cb.commit();cb.waitUntilCompleted();precondition(cb.status == .completed)
   var p=[UInt8](repeating:0,count:w*h*4);p.withUnsafeMutableBytes{out.getBytes($0.baseAddress!,bytesPerRow:w*4,from:MTLRegionMake2D(0,0,w,h),mipmapLevel:0)}
   for y in stride(from:200,to:900,by:43){for x in stride(from:200,to:1700,by:47){let r=Int(round(((Float(x)-32*t)/Float(w)+0.08)*255)),g=Int(round(((Float(y)+24*t)/Float(h)+0.08)*255));maximum=max(maximum,abs(Int(p[(y*w+x)*4+2])-r),abs(Int(p[(y*w+x)*4+1])-g))}}
  }
  precondition(maximum<=1,"Native flow-grid scale/direction mismatch: \(maximum)")
  print("PASS: all five phases, native240x135 flow-grid units ->1920x1080, +32/-24px motion, single-anchor appearance; max delta \(maximum)")
 }
}
