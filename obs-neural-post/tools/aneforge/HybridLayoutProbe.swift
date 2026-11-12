// Actual MLX buffer -> Metal pack -> ANE/Metal FFN -> Metal unpack -> MLX.
// Captured-stage replay; this does not install a backend in live OBS.
import Foundation
import Metal
import IOSurface
import MLX
import DLSSMLX

func require(_ condition: Bool, _ message: String) throws {
  if !condition { throw NSError(domain: message, code: 1) }
}
func floats(_ data: Data) -> [Float16] {
  data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
}
func median(_ values: [Double]) -> Double {
  let sorted = values.sorted(); return sorted[sorted.count / 2]
}
func compileProgram(_ specification: [String: Any]) throws -> UnsafeMutableRawPointer {
  let inputs = specification["inputs"] as! [[String: Any]]
  let owned = inputs.map { strdup($0["name"] as! String)! }
  defer { owned.forEach { free($0) } }
  let names: [UnsafePointer<CChar>?] = owned.map { UnsafePointer($0) }
  let sizes = inputs.map { $0["bytes"] as! Int }
  let outputs = specification["outputs"] as? [[String: Any]] ?? [["name": specification["output"]!, "bytes": specification["outputBytes"]!]]
  let outputOwned = outputs.map { strdup($0["name"] as! String)! }
  defer { outputOwned.forEach { free($0) } }
  let outputNames: [UnsafePointer<CChar>?] = outputOwned.map { UnsafePointer($0) }
  let outputSizes = outputs.map { $0["bytes"] as! Int }
  let result = names.withUnsafeBufferPointer { names in
    sizes.withUnsafeBufferPointer { sizes in
      outputNames.withUnsafeBufferPointer { outputNames in
        outputSizes.withUnsafeBufferPointer { outputSizes in
          dlss_ane_compile_directory_ports(specification["directory"] as! String,
            names.baseAddress, sizes.baseAddress, inputs.count,
            outputNames.baseAddress, outputSizes.baseAddress, outputs.count)
        }
      }
    }
  }
  guard let result else { throw NSError(domain: "ANE-only compilation failed", code: 1) }
  return result
}
let args = CommandLine.arguments
try require(args.count == 6 || args.count == 7, "Expected manifest, captures, weights, MLX source, output JSON, optional --output-view")
let useOutputView = args.count == 7 && args[6] == "--output-view"
func obsRunning() -> Bool {
  let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
  process.arguments = ["-x", "OBS"]; process.standardOutput = Pipe(); process.standardError = Pipe()
  guard (try? process.run()) != nil else { return false }
  process.waitUntilExit(); return process.terminationStatus == 0
}
let obsAtStart = obsRunning()
let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[1]))) as! [String: Any]
let coarseQKV = manifest["coarseQKV"] as? Bool ?? false
let expandSpec = manifest["expand"] as! [String: Any], branchSpec = manifest["branch"] as! [String: Any]
let expansion = try compileProgram(expandSpec), branch = try compileProgram(branchSpec)
defer { dlss_ane_destroy(expansion); dlss_ane_destroy(branch) }
let context = coarseQKV ? dlss_af_hybrid_create_qkv(expansion, branch,
  (expandSpec["inputs"] as! [[String: Any]])[0]["name"] as! String, expandSpec["output"] as! String,
  manifest["branchInput"] as! String, manifest["branchSkip"] as! String,
  branchSpec["output"] as! String, manifest["qkvOutput"] as! String, 640) : dlss_af_hybrid_create(expansion, branch,
  (expandSpec["inputs"] as! [[String: Any]])[0]["name"] as! String, expandSpec["output"] as! String,
  manifest["branchInput"] as! String, manifest["branchSkip"] as! String,
  branchSpec["output"] as! String, 640)
guard let context else { fatalError("Cannot bind hybrid surfaces") }
defer { dlss_af_hybrid_destroy(context) }
let device = MTLCreateSystemDefaultDevice()!
let outputBuffer = device.makeBuffer(length: 640 * 1024 * 2, options: .storageModeShared)!
let outputSurface = dlss_af_hybrid_copy_output_surface(context)!
let qkvSurface = coarseQKV ? dlss_af_hybrid_copy_qkv_surface(context) : nil
func projectedArray() -> MLXArray {
  guard let surface = qkvSurface else { fatalError("No QKV output surface") }
  return MLXArray(rawPointer: IOSurfaceGetBaseAddress(surface), [3072, 640], dtype: .float16) {
    [surface] in _ = surface
  }.transposed(1, 0)
}
func outputArray() -> MLXArray {
  if useOutputView {
    return MLXArray(rawPointer: IOSurfaceGetBaseAddress(outputSurface), [1024, 640], dtype: .float16) {
      [outputSurface] in _ = outputSurface
    }.transposed(1, 0)
  }
  return MLXArray(rawPointer: outputBuffer.contents(), [640, 1024], dtype: .float16) {
    [outputBuffer] in _ = outputBuffer
  }
}
let file = try FileHandle(forReadingFrom: URL(fileURLWithPath: args[3]))
let headerSize = try file.read(upToCount: 8)!.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
let header = try JSONSerialization.jsonObject(with: file.read(upToCount: Int(headerSize))!) as! [String: Any]
func weight(_ name: String) throws -> MLXArray {
  let entry = header[name] as! [String: Any], offsets = entry["data_offsets"] as! [Int]
  let dtype = entry["dtype"] as! String
  try require(dtype == "F16" || dtype == "F32", "Original floating-point weights required")
  try file.seek(toOffset: 8 + headerSize + UInt64(offsets[0]))
  let data = try file.read(upToCount: offsets[1] - offsets[0])!
  if dtype == "F32" {
    let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    return MLXArray(values, (entry["shape"] as! [Int])).asType(.float16)
  }
  return MLXArray(floats(data), (entry["shape"] as! [Int]))
}
let w0 = try weight("block31.layer0.weight"), w1 = try weight("block31.layer1.weight")
let cosine = try weight("block31.layer1.ffn_cos_skip")
let qkv = try weight("block31.layer2.qkv_weight")
let attentionScale = try weight("block31.layer2.attn_scale")
let attentionProjection = try weight("block31.layer4.projection_weight")
let attentionCosine = try weight("block31.layer4.attn_cos_skip")
try file.close()
let sourceText = try String(contentsOfFile: args[4], encoding: .utf8)
func shader(_ pattern: String) throws -> String {
  let regex = try NSRegularExpression(pattern: pattern, options: .dotMatchesLineSeparators)
  guard let match = regex.firstMatch(in: sourceText, range: NSRange(sourceText.startIndex..., in: sourceText)),
        let range = Range(match.range(at: 1), in: sourceText) else { fatalError("Fused shader extraction failed") }
  return String(sourceText[range])
}
let gate = MLXFast.metalKernel(name: "dlss_hybrid_layout_reference_gate",
  inputNames: ["input", "params"], outputNames: ["output"],
  source: try shader("private static let quadraticGatePublishKernel.*?source: #\"\"\"(.*?)\"\"\"#"),
  header: try shader("private static let e4m3MetalHeader = #\"\"\"(.*?)\"\"\"#"))
let params = MLXArray([UInt32(640 * 4096 / 4)])
eval(w0, w1, cosine, qkv, params, attentionScale, attentionProjection, attentionCosine)
func gpu(_ source: MLXArray) -> MLXArray {
  let expanded = matmul(source, w0)
  let published = gate([expanded, params], grid: (640 * 4096 / 4, 1, 1),
    threadGroup: (256, 1, 1), outputShapes: [expanded.shape], outputDTypes: [.float16])[0]
  return matmul(published, w1) + source * cosine
}
func referenceStage(_ source: MLXArray) -> MLXArray {
  let mid = gpu(source); return coarseQKV ? matmul(mid, qkv) : mid
}
func hybridStageOutput() -> MLXArray { coarseQKV ? projectedArray() : outputArray() }
let compiledReferenceBlock = compile { (input: MLXArray) -> MLXArray in
  MLXGlobalBlockProbe.reference(input, expansion: w0, branch: w1, ffnCosine: cosine,
    qkv: qkv, scale: attentionScale, projection: attentionProjection, attentionCosine: attentionCosine)
}
let compiledFinishBlock = compile { (inputs: [MLXArray]) -> [MLXArray] in
  [MLXGlobalBlockProbe.finish(mid: inputs[0], qkv: inputs[1], scale: attentionScale,
    projection: attentionProjection, attentionCosine: attentionCosine)]
}
func completeGPUBlock(_ source: MLXArray) -> MLXArray { compiledReferenceBlock(source) }
func completeHybridBlock() -> MLXArray {
  compiledFinishBlock([outputArray(), coarseQKV ? projectedArray() : matmul(outputArray(), qkv)])[0]
}
func execute(_ inputBuffer: MTLBuffer) throws -> [Double] {
  var timings = [Double](repeating: 0, count: 9)
  let rc = dlss_af_hybrid_execute_buffers(context,
    Unmanaged.passUnretained(inputBuffer as AnyObject).toOpaque(),
    useOutputView ? nil : Unmanaged.passUnretained(outputBuffer as AnyObject).toOpaque(), &timings)
  try require(rc == 0, "Hybrid layout execute failed: \(rc)")
  return timings
}
let captures = URL(fileURLWithPath: args[2])
let features = try FileManager.default.contentsOfDirectory(at: captures, includingPropertiesForKeys: nil)
  .map { $0.appendingPathComponent("global-input.f16") }.filter { FileManager.default.fileExists(atPath: $0.path) }
  .sorted { $0.path < $1.path }
try require(!features.isEmpty, "No captured features")
var cases = [[String: Any]](), lastSource: MLXArray?, lastBuffer: MTLBuffer?
var unpackChangedHalfWords = 0
var boundaryChecksPassed = false
for feature in features {
  let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: feature.deletingPathExtension().appendingPathExtension("json"))) as! [String: Any]
  try require(metadata["shape"] as! [Int] == [1, 20, 32, 1024], "Unexpected feature shape")
  let source = MLXArray(floats(try Data(contentsOf: feature)), [640, 1024])
  eval(source)
  try require(source.asData(access: .noCopy).strides == [1024, 1], "Input must be contiguous before no-copy wrapping")
  guard let buffer = source.asMTLBuffer(device: device, noCopy: true) else { fatalError("Cannot wrap contiguous MLX array") }
  if !boundaryChecksPassed {
    var check = [Double](repeating: 0, count: 9)
    let ptr = Unmanaged.passUnretained(buffer as AnyObject).toOpaque()
    let short = device.makeBuffer(length: 256, options: .storageModeShared)!
    let shortPtr = Unmanaged.passUnretained(short as AnyObject).toOpaque()
    try require(dlss_af_hybrid_execute_buffers(context, ptr, ptr, &check) == -2, "Aliased output must be rejected")
    try require(dlss_af_hybrid_execute_buffers(context, ptr, shortPtr, &check) == -2, "Short output must be rejected")
    try require(dlss_af_hybrid_execute_buffers(context, nil, ptr, &check) == -1, "Null input must be rejected")
    boundaryChecksPassed = true
  }
  _ = try execute(buffer)
  let actual = outputArray().asArray(Float16.self), expected = gpu(source).asArray(Float16.self)
  if !useOutputView {
    let view = MLXArray(rawPointer: IOSurfaceGetBaseAddress(outputSurface), [1024, 640], dtype: .float16) {
      [outputSurface] in _ = outputSurface
    }.transposed(1, 0).asArray(Float16.self)
    unpackChangedHalfWords += zip(actual, view).filter { $0.bitPattern != $1.bitPattern }.count
  }
  let consumed = (coarseQKV ? projectedArray() : matmul(outputArray(), qkv)).asArray(Float16.self)
  let consumedReference = matmul(gpu(source), qkv).asArray(Float16.self)
  var consumedError = 0.0, consumedSquare = 0.0
  for (a, b) in zip(consumed, consumedReference) {
    consumedError += pow(Double(a) - Double(b), 2); consumedSquare += pow(Double(b), 2)
  }
  let consumedRelative = sqrt(consumedError / max(consumedSquare, 1e-24))
  let final = completeHybridBlock().asArray(Float16.self)
  let finalReference = completeGPUBlock(source).asArray(Float16.self)
  let referenceMid = gpu(source)
  let splitReference = compiledFinishBlock([referenceMid, matmul(referenceMid, qkv)])[0].asArray(Float16.self)
  var splitError = 0.0, splitSquare = 0.0
  for (a, b) in zip(splitReference, finalReference) {
    splitError += pow(Double(a) - Double(b), 2); splitSquare += pow(Double(b), 2)
  }
  let splitRelative = sqrt(splitError / max(splitSquare, 1e-24))
  var finalError = 0.0, finalSquare = 0.0
  for (a, b) in zip(final, finalReference) {
    finalError += pow(Double(a) - Double(b), 2); finalSquare += pow(Double(b), 2)
  }
  let finalRelative = sqrt(finalError / max(finalSquare, 1e-24))
  var squareError = 0.0, squareReference = 0.0, maxError = 0.0, finite = true
  for (a, b) in zip(actual, expected) {
    finite = finite && a.isFinite
    let error = Double(a) - Double(b)
    squareError += error * error; squareReference += Double(b) * Double(b)
    maxError = max(maxError, abs(error))
  }
  let relative = sqrt(squareError / max(squareReference, 1e-24))
  cases.append(["feature": feature.path, "relativeL2": relative, "finite": finite,
    "maxAbsError": maxError, "nextProjectionRelativeL2": consumedRelative,
    "completeBlockRelativeL2": finalRelative, "gpuSplitReferenceRelativeL2": splitRelative,
    "passed": finite && relative <= 0.01 && consumedRelative.isFinite && consumedRelative <= 0.01 && finalRelative.isFinite && finalRelative <= 0.01 && splitRelative == 0])
  lastSource = source; lastBuffer = buffer
}
let source = lastSource!, inputBuffer = lastBuffer!
var rows = [[Double]](), gpuTimes = [Double](), callerTimes = [Double]()
for _ in 0..<5 { _ = try execute(inputBuffer); eval(referenceStage(source)) }
for _ in 0..<50 {
  let start = ProcessInfo.processInfo.systemUptime
  // Keep the array immutable and alive for its no-copy buffer and every read.
  let timings = try withExtendedLifetime(source) { try execute(inputBuffer) }
  let result = hybridStageOutput(); eval(result)
  callerTimes.append((ProcessInfo.processInfo.systemUptime - start) * 1000)
  rows.append(timings)
  let began = ProcessInfo.processInfo.systemUptime; eval(referenceStage(source))
  gpuTimes.append((ProcessInfo.processInfo.systemUptime - began) * 1000)
}
// Include the following QKV projection so a strided output is tested by a
// real MLX consumer, not just a CPU oracle read. These are interleaved pairs.
var hybridConsumerTimes = [Double](), gpuConsumerTimes = [Double]()
for _ in 0..<5 {
  _ = try execute(inputBuffer); eval(completeHybridBlock()); eval(completeGPUBlock(source))
}
for _ in 0..<30 {
  let start = ProcessInfo.processInfo.systemUptime
  guard let wrapped = source.asMTLBuffer(device: device, noCopy: true) else { fatalError("MLX buffer wrap failed") }
  _ = try withExtendedLifetime(source) { try execute(wrapped) }
  eval(completeHybridBlock())
  hybridConsumerTimes.append((ProcessInfo.processInfo.systemUptime - start) * 1000)
  let began = ProcessInfo.processInfo.systemUptime; eval(completeGPUBlock(source))
  gpuConsumerTimes.append((ProcessInfo.processInfo.systemUptime - began) * 1000)
}
let labels = ["completeWallMS", "packWallMS", "innerWallMS", "unpackWallMS", "packGPUKernelMS",
  "gateGPUKernelMS", "unpackGPUKernelMS", "expandANEWallMS", "branchANEWallMS"]
let medians = Dictionary(uniqueKeysWithValues: labels.enumerated().map { index, name in (name, median(rows.map { $0[index] })) })
let report: [String: Any] = ["scope": coarseQKV ? "GPU pack -> ANE expansion -> Metal gate -> ANE branch + residual + QKV -> original GPU attention" : "MLX buffer -> GPU pack -> hybrid FFN -> original GPU attention",
  "completeGlobalBlockHybridMedianMS": median(hybridConsumerTimes),
  "completeGlobalBlockGPUReferenceMedianMS": median(gpuConsumerTimes), "coarseQKV": coarseQKV,
  "unpackParityChecked": !useOutputView, "unpackChangedHalfWords": unpackChangedHalfWords,
  "bufferBoundaryChecksPassed": boundaryChecksPassed,
  "weightINT8": manifest["weightINT8"]!, "activationPrecision": "FP16", "cases": cases,
  "allPassed": cases.allSatisfy { $0["passed"] as! Bool }, "medians": medians,
  "swiftCallerMedianMS": median(callerTimes), "fusedGateGPUReferenceMedianMS": median(gpuTimes),
  "hotLoopCPUTensorCopies": 0, "layoutGPUTransforms": useOutputView ? 1 : 2,
  "stridedOutputView": useOutputView, "obsRunningAtStart": obsAtStart, "obsRunningAtEnd": obsRunning(), "productionInstalled": false,
  "limitations": "Captured-stage replay; excludes full-model scheduling and fresh GPU producer cost; production attention and residual measured as complete global block"]
let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
try json.write(to: URL(fileURLWithPath: args[5])); print(String(decoding: json, as: UTF8.self))
