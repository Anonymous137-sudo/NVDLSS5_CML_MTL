import Foundation
import CoreML
import MLX

/// A single prediction client; the pipeline admits one frame at a time.
/// Compilation runs separately so live GPU control frames remain available.
final class LiveANEIsland: @unchecked Sendable {
 private let lock = NSLock()
 private let globalOnly=CommandLine.arguments.contains("--split-global")
 private var shape:[Int] {globalOnly ? [1,1024,12,20] : [1,512,24,40]}
 private var loaded: MLModel?
 private var state: [String:Any] = ["phase":"compiling", "computeUnits":"cpuAndNeuralEngine", "scope":"HT blocks23...47"]
 init(url:URL) {
  Task.detached(priority:.userInitiated) { [self] in
   do {
    let start=ProcessInfo.processInfo.systemUptime
    update(["scope":globalOnly ? "HT blocks31...38 only" : "HT blocks23...47"])
    update(["phase":"waitingForPackage"])
    while !FileManager.default.fileExists(atPath:url.path) {
     if ProcessInfo.processInfo.systemUptime-start>600 {throw NSError(domain:"StageIsland",code:6,userInfo:[NSLocalizedDescriptionKey:"Export package did not arrive in600 seconds"])}
     try await Task.sleep(nanoseconds:1_000_000_000)
    }
    update(["phase":"compiling"])
    let compiled = url.pathExtension == "mlmodelc" ? url : try await MLModel.compileModel(at:url)
    update(["phase":"loadingDeviceModel", "portableCompileSeconds":ProcessInfo.processInfo.systemUptime-start])
    let configuration=MLModelConfiguration();configuration.computeUnits = .cpuAndNeuralEngine
    let model=try MLModel(contentsOf:compiled,configuration:configuration)
    guard let metadata=model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String:String],
      metadata["stage_contract"] == (globalOnly ? "ht-720-global-31-38-v1" : "ht-720-center-23-47-v1"),
      model.modelDescription.inputDescriptionsByName["features"]?.multiArrayConstraint?.shape.map(\.intValue)==shape else {
      throw NSError(domain:"StageIsland",code:1,userInfo:[NSLocalizedDescriptionKey:"Wrong stage input contract"])
    }
    setModel(model)
    update(["phase":"ready", "loadSeconds":ProcessInfo.processInfo.systemUptime-start])
    // Inspect placement while capture and predictions continue; counts do not
    // claim a physical utilization trace. The compiled directory stays alive.
    do {
     let plan=try await MLComputePlan.load(contentsOf:compiled,configuration:configuration)
     var counts:[String:Int]=[:]
     func walk(_ block:MLModelStructure.Program.Block) {
      for op in block.operations {
       var name="unreported"
       if let usage=plan.deviceUsage(for:op) {
        switch usage.preferred {case .cpu:name="CPU";case .gpu:name="GPU";case .neuralEngine:name="ANE";@unknown default:name="unknown"}
       }
       counts[name,default:0]+=1;op.blocks.forEach(walk)
      }
     }
     if case let .program(program)=plan.modelStructure { for function in program.functions.values {walk(function.block)} }
     update(["preferredOperationCounts":counts,"placementNote":"Compiler preference, not Instruments utilization."])
    } catch { update(["computePlanError":String(describing:error)]) }
   } catch { update(["phase":"failed", "error":String(describing:error)]) }
  }
 }
 private func update(_ values:[String:Any]) {lock.lock();state.merge(values){_,new in new};lock.unlock()}
 private func setModel(_ model:MLModel) {lock.lock();loaded=model;lock.unlock()}
 private func model()->MLModel? {lock.lock();defer{lock.unlock()};return loaded}
 var ready:Bool {model() != nil}
 var report:[String:Any] {lock.lock();defer{lock.unlock()};return state}
 func predict(_ input:MLXArray) async throws -> (MLXArray,Double) {
  guard let model=model() else {throw NSError(domain:"StageIsland",code:2)}
  let start=ProcessInfo.processInfo.systemUptime
  let nchw=input.transposed(0,3,1,2).asType(.float16);eval(nchw)
  let host=nchw.asArray(Float16.self)
  let shape=self.shape,c=shape[1],h=shape[2],w=shape[3]
  let array=try MLMultiArray(shape:shape.map{NSNumber(value:$0)},dataType:.float16)
  let strides=array.strides.map(\.intValue)
  guard strides == [c*h*w,h*w,w,1] else {throw NSError(domain:"StageIsland",code:3)}
  _ = host.withUnsafeBytes { memcpy(array.dataPointer,$0.baseAddress!,$0.count) }
  let provider=try MLDictionaryFeatureProvider(dictionary:["features":MLFeatureValue(multiArray:array)])
  let inferenceStart=ProcessInfo.processInfo.systemUptime
  let output=try await model.prediction(from:provider)
  let inferenceMS=(ProcessInfo.processInfo.systemUptime-inferenceStart)*1000
  guard let result=output.featureValue(for:"restored")?.multiArrayValue,
    result.dataType == .float16,result.shape.map(\.intValue)==shape else {throw NSError(domain:"StageIsland",code:4)}
  let outStrides=result.strides.map(\.intValue)
  let ptr=result.dataPointer.assumingMemoryBound(to:Float16.self)
  var values=[Float16](repeating:0,count:c*h*w)
  for channel in 0..<c {for y in 0..<h {for x in 0..<w {
   let value=ptr[channel*outStrides[1]+y*outStrides[2]+x*outStrides[3]]
   guard value.isFinite else {throw NSError(domain:"StageIsland",code:5,userInfo:[NSLocalizedDescriptionKey:"Nonfinite ANE stage output"])}
   values[(channel*h+y)*w+x]=value
  }}}
  let restored=MLXArray(values,shape).transposed(0,2,3,1);eval(restored)
  let elapsed=(ProcessInfo.processInfo.systemUptime-start)*1000
  update(["lastPredictionMS":inferenceMS,"lastTransferAndPredictionMS":elapsed])
  return (restored,elapsed)
 }
}
