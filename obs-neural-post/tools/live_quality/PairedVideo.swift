import AVFoundation
import CoreVideo
import Foundation

/// Optional bounded moving-scene evidence. Append only matched processed/source
/// pairs using original source PTS. Recording overhead stays visible in reports.
final class PairedVideo {
 private let queue=DispatchQueue(label:"stage-split.paired-video")
 private let slot=DispatchSemaphore(value:1),lock=NSLock()
 private var writers:[AVAssetWriter]=[],inputs:[AVAssetWriterInput]=[],adaptors:[AVAssetWriterInputPixelBufferAdaptor]=[]
 private var start:Double?,count=0,dropped=0,finished=false,error=""
 private let destination:URL,duration:Double
 init(destination:URL,duration:Double){self.destination=destination;self.duration=duration}
 var report:[String:Any]{lock.lock();defer{lock.unlock()};return ["enabled":true,"pairedFrames":count,"droppedPairs":dropped,"finished":finished,"error":error,"durationLimitSeconds":duration,"note":"Original source timestamps; fresh processed frames only. Encoding overhead affects live measurements."]}
 func append(source:CVPixelBuffer,output:CVPixelBuffer,pts:Double){
  lock.lock();let done=finished;lock.unlock();if done{return}
  guard slot.wait(timeout:.now()) == .success else {lock.lock();dropped+=1;lock.unlock();return}
  queue.async { [self,source,output] in
   defer{slot.signal()}
   do {
    if start==nil {
     let outputName=CVPixelBufferGetHeight(output)==1080 ? "processed-1080p" : "processed-4k"
     for (name,buffer,bitrate) in [("source",source,3_000_000),(outputName,output,16_000_000)] {
      let url=destination.appendingPathComponent("live-\(name).mov")
      let writer=try AVAssetWriter(outputURL:url,fileType:.mov)
      let settings:[String:Any]=[AVVideoCodecKey:AVVideoCodecType.hevc,AVVideoWidthKey:CVPixelBufferGetWidth(buffer),AVVideoHeightKey:CVPixelBufferGetHeight(buffer),AVVideoCompressionPropertiesKey:[AVVideoAverageBitRateKey:bitrate]]
      let input=AVAssetWriterInput(mediaType:.video,outputSettings:settings);input.expectsMediaDataInRealTime=true
      let adaptor=AVAssetWriterInputPixelBufferAdaptor(assetWriterInput:input,sourcePixelBufferAttributes:nil)
      guard writer.canAdd(input) else {throw NSError(domain:"PairedVideo",code:1)}
      writer.add(input);guard writer.startWriting() else {throw writer.error ?? NSError(domain:"PairedVideo",code:2)}
      writer.startSession(atSourceTime:.zero);writers.append(writer);inputs.append(input);adaptors.append(adaptor)
     }
     start=pts
    }
    let elapsed=pts-start!
    if elapsed>=duration {
     inputs.forEach{$0.markAsFinished()}
     for writer in writers {writer.finishWriting{}}
     lock.lock();finished=true;lock.unlock();return
    }
    guard inputs.allSatisfy(\.isReadyForMoreMediaData) else {lock.lock();dropped+=1;lock.unlock();return}
    let time=CMTime(seconds:max(0,elapsed),preferredTimescale:60000)
    guard adaptors[0].append(source,withPresentationTime:time),adaptors[1].append(output,withPresentationTime:time) else {throw NSError(domain:"PairedVideo",code:3)}
    lock.lock();count+=1;lock.unlock()
   } catch {lock.lock();self.error=String(describing:error);finished=true;lock.unlock()}
  }
 }
}
