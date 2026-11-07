// Full recovered head diagnostic, deliberately excludes the GPU from Core ML.
// Inputs are the existing native-reference NHWC float32 feature tensor.
import CoreML
import Foundation

@main struct FullModelANEAudit {
    static func main() async throws {
        guard CommandLine.arguments.count == 4 || (CommandLine.arguments.count == 5 && CommandLine.arguments[4] == "--pre-block") else {
            fatalError("MODEL.mlpackage FEATURES_NHWC.f32 REPORT.json [--pre-block]")
        }
        let preBlock = CommandLine.arguments.count == 5
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let reportURL = URL(fileURLWithPath: CommandLine.arguments[3])
        var report: [String: Any] = ["configuration": "cpuAndNeuralEngine", "model": source.path,
            "scope": preBlock ? "input adapter and block0 only; not full appearance inference" : "full recovered 71-block head",
            "note": "Preferred placement is not an Instruments utilization trace. Timings exclude capture, preparation, GPU reconstruction, presentation and OBS."]
        func save(_ stage: String) throws {
            report["stage"] = stage
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: reportURL, options: .atomic)
        }
        do {
            try save("compiling")
            let start = ProcessInfo.processInfo.systemUptime
            let compiled = source.pathExtension == "mlmodelc" ? source : try await MLModel.compileModel(at: source)
            defer { if compiled != source { try? FileManager.default.removeItem(at: compiled) } }
            report["compileSeconds"] = ProcessInfo.processInfo.systemUptime - start
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            try save("loadingComputePlan")
            let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
            var counts: [String: Int] = [:], byOperator: [String: [String: Int]] = [:]
            var estimatedCosts: [String: Double] = [:]
            func walk(_ block: MLModelStructure.Program.Block) {
                for op in block.operations {
                    let device: String
                    if let usage = plan.deviceUsage(for: op) {
                        switch usage.preferred {
                        case .cpu: device = "CPU"
                        case .gpu: device = "GPU"
                        case .neuralEngine: device = "ANE"
                        @unknown default: device = "unknown"
                        }
                    } else { device = "unreported" }
                    counts[device, default: 0] += 1
                    byOperator[op.operatorName, default: [:]][device, default: 0] += 1
                    if let cost = plan.estimatedCost(of: op) {
                        estimatedCosts[op.operatorName, default: 0] += cost.weight
                    }
                    op.blocks.forEach(walk)
                }
            }
            if case let .program(program) = plan.modelStructure {
                for function in program.functions.values { walk(function.block) }
            }
            report["preferredOperationCounts"] = counts
            report["operatorsByPreferredDevice"] = byOperator
            report["estimatedRelativeCostByOperator"] = estimatedCosts
            try save("loadingModel")
            let model = try MLModel(contentsOf: compiled, configuration: configuration)
            guard let input = model.modelDescription.inputDescriptionsByName.first(where: { $0.value.type == .multiArray }),
                  let constraint = input.value.multiArrayConstraint else { fatalError("Tensor input required") }
            let shape = constraint.shape.map(\.intValue)
            guard shape == [1, 16, 768, 1280] else { fatalError("Expected full 720p padded graph, got \(shape)") }
            let array = try MLMultiArray(shape: constraint.shape, dataType: .float32)
            let bytes = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2]))
            guard bytes.count == array.count * 4 else { fatalError("Feature byte count mismatch") }
            let strides = array.strides.map(\.intValue)
            bytes.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float.self)
                let dst = array.dataPointer.assumingMemoryBound(to: Float.self)
                for c in 0..<16 { for y in 0..<768 { for x in 0..<1280 {
                    dst[c * strides[1] + y * strides[2] + x * strides[3]] = src[(y * 1280 + x) * 16 + c]
                } } }
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: [input.key: MLFeatureValue(multiArray: array)])
            var times: [Double] = []
            for iteration in 0..<8 {
                try save("predicting-\(iteration)")
                let start = ProcessInfo.processInfo.systemUptime
                let output = try await model.prediction(from: provider)
                let ms = (ProcessInfo.processInfo.systemUptime - start) * 1000
                let tooSlowForInteractiveTrial = !preBlock && iteration > 0 && ms > 1000
                if iteration > 0 { times.append(ms) } else { report["firstPredictionMS"] = ms }
                if iteration == 7 || tooSlowForInteractiveTrial, let head = output.featureValue(for: "restored")?.multiArrayValue {
                    let s = head.shape.map(\.intValue), t = head.strides.map(\.intValue)
                    guard head.dataType == .float32, s == [1, preBlock ? 32 : 4, 768, 1280] else { fatalError("Unexpected head contract") }
                    report["outputShapeNCHW"] = s
                    if preBlock {
                        let values = head.dataPointer.assumingMemoryBound(to: Float.self)
                        var finite = true
                        for c in 0..<32 { for y in 0..<768 { for x in 0..<1280 {
                            finite = finite && values[c * t[1] + y * t[2] + x * t[3]].isFinite
                        } } }
                        report["outputFinite"] = finite
                        report["warmPredictionMS"] = times
                        if !finite { fatalError("Nonfinite pre-block output") }
                        continue
                    }
                    let src = head.dataPointer.assumingMemoryBound(to: Float.self)
                    var nhwc = [Float](repeating: 0, count: head.count)
                    for y in 0..<768 { for x in 0..<1280 { for c in 0..<4 {
                        nhwc[(y * 1280 + x) * 4 + c] = src[c * t[1] + y * t[2] + x * t[3]]
                    } } }
                    guard nhwc.allSatisfy(\.isFinite) else { fatalError("Nonfinite output") }
                    try nhwc.withUnsafeBytes { try Data($0).write(to: reportURL.deletingPathExtension().appendingPathExtension("head.f32")) }
                }
                report["warmPredictionMS"] = times
                if tooSlowForInteractiveTrial {
                    report["earlyStopReason"] = "Warm prediction exceeds one second; retain output for parity but avoid seven further unplayable repeats."
                    break
                }
            }
            report["medianPredictionMS"] = times.sorted()[times.count / 2]
            try save("complete")
        } catch {
            report["error"] = String(describing: error)
            try? save("failed")
            throw error
        }
    }
}
