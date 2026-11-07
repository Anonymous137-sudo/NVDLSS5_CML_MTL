import Foundation
import VideoToolbox
import CoreVideo
import CoreMedia
@main struct FlowProbe {
 static func main()throws {
  guard #available(macOS 15.4,*) else{return}
  let w=1920,h=1080,shift=Int(CommandLine.arguments.dropFirst().first ?? "32")!,shiftY=Int(CommandLine.arguments.dropFirst(2).first ?? "0")!
  let c=VTOpticalFlowConfiguration(frameWidth:w,frameHeight:h,qualityPrioritization:.normal,revision:.revision1)!
  let p=VTFrameProcessor();try p.startSession(configuration:c);defer{p.endSession()}
  func image(_ moving:Bool)->CVPixelBuffer {
   var p:CVPixelBuffer?;precondition(CVPixelBufferCreate(nil,w,h,kCVPixelFormatType_64RGBAHalf,[kCVPixelBufferIOSurfacePropertiesKey:[:]] as CFDictionary,&p)==kCVReturnSuccess)
   CVPixelBufferLockBaseAddress(p!,[]);let b=CVPixelBufferGetBaseAddress(p!)!.assumingMemoryBound(to:Float16.self),row=CVPixelBufferGetBytesPerRow(p!)/2
   for y in 0..<h {for x in 0..<w {
    let sx=max(0,min(w-1,x-(moving ? shift:0))),sy=max(0,min(h-1,y-(moving ? shiftY:0))),xx=sx/12,yy=sy/12,hash=(xx*137+yy*73+xx*yy*17)%251
    let v=Float16(Float(hash)/300+0.05),i=y*row+x*4;b[i]=v;b[i+1]=v;b[i+2]=v;b[i+3]=1
   }}
   CVPixelBufferUnlockBaseAddress(p!,[]);return p!
  }
  func flow()->CVPixelBuffer {
   var p:CVPixelBuffer?;let a=c.destinationPixelBufferAttributes
   precondition(CVPixelBufferCreate(nil,a[kCVPixelBufferWidthKey as String] as! Int,a[kCVPixelBufferHeightKey as String] as! Int,kCVPixelFormatType_TwoComponent16Half,a as CFDictionary,&p)==kCVReturnSuccess);return p!
  }
  let a=image(false),b=image(true),f=flow(),r=flow(),flow=VTFrameProcessorOpticalFlow(forwardFlow:f,backwardFlow:r)!
  let params=VTOpticalFlowParameters(sourceFrame:VTFrameProcessorFrame(buffer:a,presentationTimeStamp:CMTime(value:0,timescale:60))!,nextFrame:VTFrameProcessorFrame(buffer:b,presentationTimeStamp:CMTime(value:6,timescale:60))!,submissionMode:.random,destinationOpticalFlow:flow)!
  let done=DispatchSemaphore(value:0);var error:Error?
  p.process(parameters:params){_,e in error=e;done.signal()};precondition(done.wait(timeout:.now()+30) == .success);if let error{throw error}
  for (name,buffer) in [("forward",f),("backward",r)] {
   CVPixelBufferLockBaseAddress(buffer,.readOnly);defer{CVPixelBufferUnlockBaseAddress(buffer,.readOnly)}
   let p=CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to:Float16.self),row=CVPixelBufferGetBytesPerRow(buffer)/2,fw=CVPixelBufferGetWidth(buffer),fh=CVPixelBufferGetHeight(buffer)
   var xs:[Float]=[],ys:[Float]=[]
   for y in stride(from:fh/4,to:fh*3/4,by:7){for x in stride(from:fw/4,to:fw*3/4,by:7){xs.append(Float(p[y*row+x*2]));ys.append(Float(p[y*row+x*2+1]))}}
   xs.sort();ys.sort();print(name,"grid",fw,fh,"median",xs[xs.count/2],ys[ys.count/2])
  }
 }
}
