import Foundation
import CoreVideo
import CoreGraphics
import ImageIO
import Metal

/// Opt-in, bounded live-pair evidence. PNG encoding never runs on inference,
/// capture or OBS render threads. The capture pool buffer is never retained.
final class LiveFrameDiagnostics {
 private let queue=DispatchQueue(label:"obs.nr.pair-evidence",qos:.utility),lock=NSLock()
 private var busy=false,lastEpoch:UInt64=UInt64.max,count=0,lastTime=0.0,total=0
 private var previous:ProcessedFrame?
 private let directory=ProcessInfo.processInfo.environment["DLSS_NR_PAIR_DUMP"]
 private struct Image {let data:Data,width:Int,height:Int,stride:Int}
 private func copy(_ p:CVPixelBuffer)->Image? {
  guard CVPixelBufferGetPixelFormatType(p)==kCVPixelFormatType_32BGRA,CVPixelBufferLockBaseAddress(p,.readOnly)==kCVReturnSuccess else{return nil}
  defer{CVPixelBufferUnlockBaseAddress(p,.readOnly)}
  guard let base=CVPixelBufferGetBaseAddress(p) else{return nil}
  let w=CVPixelBufferGetWidth(p),h=CVPixelBufferGetHeight(p),row=CVPixelBufferGetBytesPerRow(p)
  return Image(data:Data(bytes:base,count:row*h),width:w,height:h,stride:row)
 }
 private func copy(_ t:MTLTexture)->Image? {
  guard t.pixelFormat == .bgra8Unorm else{return nil}
  let row=t.width*4;var data=Data(count:row*t.height)
  data.withUnsafeMutableBytes{t.getBytes($0.baseAddress!,bytesPerRow:row,from:MTLRegionMake2D(0,0,t.width,t.height),mipmapLevel:0)}
  return Image(data:data,width:t.width,height:t.height,stride:row)
 }
 func submit(_ input:CVPixelBuffer,_ output:ProcessedFrame,_ config:NRConfiguration,_ epoch:UInt64,_ generated:[ProcessedFrame],_ guides:[CaptureGuide]) {
  guard let directory else{return}
  let now=ProcessInfo.processInfo.systemUptime
  lock.lock();if lastEpoch != epoch {lastEpoch=epoch;count=0;lastTime=0;previous=nil}
  let prior=previous;previous=output
  guard (config.frameGenerationFactor<=1 || !generated.isEmpty),!busy,count<3,total<24,now-lastTime>1 else{lock.unlock();return};busy=true;count+=1;total+=1;lastTime=now;lock.unlock()
  guard let raw=copy(input),let nr=copy(output.pixels) else{lock.lock();busy=false;lock.unlock();return}
  var images:[(String,Image)]=[("capture",raw),("base-output",nr)]
  if let prior,let image=copy(prior.pixels) {images.append(("previous-base",image));if let guide=prior.motionGuide,let raw=copy(guide.texture){images.append(("previous-capture",raw))}}
  let intermediate=generated.isEmpty ? nil:generated[generated.count/2]
  if let intermediate,let image=copy(intermediate.pixels) {
   images.append(("mfg-intermediate",image))
   if let guide=guides.min(by:{abs($0.seconds-intermediate.sourceSeconds)<abs($1.seconds-intermediate.sourceSeconds)}),abs(guide.seconds-intermediate.sourceSeconds)<0.0001,let image=copy(guide.texture){images.append(("actual-intermediate-capture",image))}
  }
  let sequence=output.sequence,source=output.sourceSeconds,stages=output.stages
  let intermediateTime=intermediate?.sourceSeconds,intermediatePhase=intermediate?.phase
  queue.async {[self] in
   defer{lock.lock();busy=false;lock.unlock()}
   do {
    let dir=URL(fileURLWithPath:directory).appendingPathComponent("epoch-\(epoch)-seq-\(sequence)")
    try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true)
    for (name,image) in images {
     guard let provider=CGDataProvider(data:image.data as CFData),let cg=CGImage(width:image.width,height:image.height,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:image.stride,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGBitmapInfo.byteOrder32Little.rawValue|CGImageAlphaInfo.noneSkipFirst.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent),let destination=CGImageDestinationCreateWithURL(dir.appendingPathComponent(name+".png") as CFURL,"public.png" as CFString,1,nil) else{continue}
     CGImageDestinationAddImage(destination,cg,nil);CGImageDestinationFinalize(destination)
    }
    let settings=try JSONSerialization.jsonObject(with:JSONEncoder().encode(config))
    var metadata:[String:Any]=["sequence":sequence,"sourceSeconds":source,"settings":settings,"stages":stages,"input":"exact720p capture for this base output; no interpolated frame"]
    if let intermediateTime,let intermediatePhase {metadata["intermediateSourceSeconds"]=intermediateTime;metadata["intermediatePhase"]=intermediatePhase}
    try JSONSerialization.data(withJSONObject:metadata,options:[.prettyPrinted,.sortedKeys]).write(to:dir.appendingPathComponent("pair.json"))
   } catch {NSLog("NR pair evidence: %@",String(describing:error))}
  }
 }
}
