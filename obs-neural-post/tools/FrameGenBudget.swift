import Foundation
import DLSSMLX
import MLX
@main struct FrameGenBudget {
 static func main() throws {
  guard CommandLine.arguments.count==3 else { fatalError("WEIGHTS INPUT_DIRECTORY") }
  let base=URL(fileURLWithPath:CommandLine.arguments[2])
  func image(_ name:String,_ h:Int,_ w:Int)throws->MLXArray {
   MLXArray(try Data(contentsOf:base.appendingPathComponent(name)),[1,h,w,3],dtype:.float32).asType(.float16)
  }
  let a=try image("a720.f32",720,1280),b=try image("b720.f32",720,1280),a4=try image("a4k.f32",2160,3840),b4=try image("b4k.f32",2160,3840)
  eval(a,b,a4,b4)
  let gen=try FrameGenerator(weightsURL:URL(fileURLWithPath:CommandLine.arguments[1]),precision:.float16)
  var times:[Double]=[],last:MLXArray?
  for i in 0..<5 {
   let start=ProcessInfo.processInfo.systemUptime
   let coarse=gen.synthesize(a,b,phase:0.5)
   let out=FrameGenerator.compose(a4,b4,coarse:coarse,scale:6,sigmoidMask:true)
   eval(out);if i>0 { times.append((ProcessInfo.processInfo.systemUptime-start)*1000) };last=out
  }
  try last!.asType(.float32).asData(access:.copy).data.write(to:base.appendingPathComponent("generated4k.f32"))
  print(String(data:try JSONSerialization.data(withJSONObject:["method":"720p learned FG flow/mask, full-4K warp/composition, scale=6","warmSamples":times.count,"timesMS":times,"medianMS":times.sorted()[times.count/2],"note":"Synthetic 8-source-pixel translation; no SR/appearance/OBS cost included. Cross-resolution synthesis quality requires separate validation."],options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
 }
}
