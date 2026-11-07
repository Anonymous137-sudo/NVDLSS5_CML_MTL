import Foundation
import DLSSCore
import DLSSMLX
import MLX

@main struct NativeQualityPipeline {
 static func main() async throws {
  guard [7,8].contains(CommandLine.arguments.count) else { fatalError("MODEL SR HIGH720.f32 LOW.f32 WIDTH HEIGHT") }
  let lw=Int(CommandLine.arguments[5])!,lh=Int(CommandLine.arguments[6])!
  let geometry=try NeuralRenderingNetworkGeometry(outputWidth:lw,outputHeight:lh,networkWidth:((max(128,lw)+63)/64)*64,networkHeight:((max(128,lh)+63)/64)*64)
  let lowData=try Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[4]))
  let low=try HostTensor(descriptor:TensorDescriptor(name:"color",shape:[1,lh,lw,3],dataType:.float32,layout:.nhwc),bytes:lowData)
  let high=MLXArray(try Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[3])),[1,720,1280,3],dtype:.float32).asType(.float16)
  let originalLow=MLXArray(lowData,[1,lh,lw,3],dtype:.float32).asType(.float16)
  let model=try MLXNeuralRenderer(packageURL:URL(fileURLWithPath:CommandLine.arguments[1]),executionMode:.metalFused,computePrecision:.float16)
  let sr=try CompactSR(url:URL(fileURLWithPath:CommandLine.arguments[2]))
  let generator=CommandLine.arguments.count==8 ? try FrameGenerator(weightsURL:URL(fileURLWithPath:CommandLine.arguments[7]),precision:.float16) : nil
  var previous720:MLXArray?,previous4K:MLXArray?,generated:MLXArray?
  let guide=MLXFast.metalKernel(name:"quality_low_delta",inputNames:["original","styled","high"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;if(p>=1280*720) return;
  uint x=p%1280,y=p/1280;
  float2 q=clamp((float2(x,y)+.5f)*float2(LW/1280.f,LH/720.f)-.5f,float2(0),float2(LW-1,LH-1));
  uint2 a=uint2(q),b=min(a+1,uint2(LW-1,LH-1));float2 f=fract(q);
  bool hud=x<333 && y>=504;
  for(uint c=0;c<3;c++) {
   uint i0=(a.y*LW+a.x)*3+c,i1=(a.y*LW+b.x)*3+c,i2=(b.y*LW+a.x)*3+c,i3=(b.y*LW+b.x)*3+c;
   float d=mix(mix(float(styled[i0])-float(original[i0]),float(styled[i1])-float(original[i1]),f.x),mix(float(styled[i2])-float(original[i2]),float(styled[i3])-float(original[i3]),f.x),f.y);
   out[p*3+c]=clamp(float(high[p*3+c])+(hud ? 0.f : d),0.f,1.f);
  }
  """)
  let pack=MLXFast.metalKernel(name:"quality_sr_pack",inputNames:["rgb","bias"],outputNames:["out"],source:"""
  uint p=thread_position_in_grid.x;if(p>=3840*2160) return;
  uint x=p%3840,y=p/3840,sub=(y%6)*6+x%6;
  uint i=((y/6)*640+x/6)*108+sub;
  for(uint c=0;c<3;c++) out[p*3+c]=uchar(clamp(round(float(rgb[i+c*36]+bias[sub+c*36])*255.f),0.f,255.f));
  """)
  eval(high,originalLow)
  var times:[Double]=[],forward:[Double]=[],last:MLXArray?
  for i in 0..<12 {
   let begin=ProcessInfo.processInfo.systemUptime
   let features=try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(from:low,noiseFrameIndex:0,geometry:geometry,normalizedStyle:0,localToneStrength:1,localStructureStrength:1,automaticMask:nil,controlMask:nil)
   let result=try await model.render(NeuralRenderRequest(sequenceID:UInt64(i),inputs:[features]))
   let head=try geometry.cropOutput(result.output(named:"color")!)
   let styled=try NeuralRenderingFirstFramePostprocessor.compose(head:head,over:low,controlMask:nil,intensity:1)
   guard styled.bytes.withUnsafeBytes({ $0.bindMemory(to:Float.self).allSatisfy { $0.isFinite } }) else { throw NSError(domain:"QualityPipeline",code:1) }
   let appearance=MLXArray(styled.bytes,[1,lh,lw,3],dtype:.float32).asType(.float16)
   let image=guide([originalLow,appearance,high],template:[("LW",lw),("LH",lh)],grid:(1280*720,1,1),threadGroup:(256,1,1),outputShapes:[[1,720,1280,3]],outputDTypes:[.float16])[0]
   let rgb=sr.head(image)
   let output=pack([rgb,sr.outputBias],grid:(3840*2160,1,1),threadGroup:(256,1,1),outputShapes:[[2160,3840,3]],outputDTypes:[.uint8])[0]
   if let generator,let previous720,let previous4K {
    let coarse=generator.synthesize(previous720,image,phase:0.5)
    let current=output.reshaped(1,2160,3840,3).asType(.float16)/255
    let intermediate=FrameGenerator.compose(previous4K,current,coarse:coarse,scale:6,sigmoidMask:true)
    generated=(intermediate*255+0.5).asType(.uint8)
    eval(output,generated!)
    withExtendedLifetime((previous720,previous4K)) {}
   } else { eval(output) }
   if generator != nil { previous720=image;previous4K=output.reshaped(1,2160,3840,3).asType(.float16)/255;eval(previous4K!) }
   last=output
   if i>2 { times.append((ProcessInfo.processInfo.systemUptime-begin)*1000);forward.append(Double(result.timing.executionNanoseconds)/1e6) }
  }
  let sorted=times.sorted();let dir=URL(fileURLWithPath:CommandLine.arguments[4]).deletingLastPathComponent()
  try last!.asData(access:.copy).data.write(to:dir.appendingPathComponent("native-quality-4k.rgb8"))
  if let generated { try generated.asData(access:.copy).data.write(to:dir.appendingPathComponent("native-generated-4k.rgb8")) }
  let report:[String:Any]=["logical":[lw,lh],"network":[geometry.networkWidth,geometry.networkHeight],"warmSamples":times.count,"timesMS":times,"medianMS":sorted[sorted.count/2],"maxMS":sorted.last!,"meanCompletionFPS":1000/(times.reduce(0,+)/Double(times.count)),"NRForwardMS":forward,"framegen":generator != nil,"intermediate_frames_per_base":generator == nil ? 0 : 1,"note":"Native MLX0.31 full-model single-pass low-res appearance + original720 detail guide + RT4KSR3x + fused RGB8 output. Includes CPU feature preparation, low-res readback/composition. Cached static inputs; FG, when enabled, is an identity-pair workload, not motion validation. Excludes downsample/capture, final output readback/presentation and OBS. Nine timed frames after three warmups are not a sustained-FPS proof."]
  print(String(data:try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
 }
}
