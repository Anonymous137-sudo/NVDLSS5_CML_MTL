// Read-only model audit. Standalone diagnostic; does not change the OBS runtime.
// Compile: xcrun swiftc -parse-as-library -target arm64-apple-macosx15.0
//   tools/CoreMLAudit.swift -o build/coreml-audit
import CoreML
import CoreVideo
import Foundation
import CoreImage

@main struct CoreMLAudit {
    static func main() async throws {
        guard CommandLine.arguments.count >= 2 else { fatalError("Pass a model .mlpackage or .mlmodelc, optionally --image=PATH") }
        let imagePath = CommandLine.arguments.first(where:{$0.hasPrefix("--image=")}).map { String($0.dropFirst(8)) }
        let source = URL(fileURLWithPath:CommandLine.arguments[1])
        let compiled = source.pathExtension == "mlmodelc" ? source : try await MLModel.compileModel(at:source)
        defer { if source != compiled { try? FileManager.default.removeItem(at:compiled) } }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let model = try MLModel(contentsOf:compiled,configuration:configuration)
        guard let input = model.modelDescription.inputDescriptionsByName.first(where:{$0.value.type == .image}),
              let constraint = input.value.imageConstraint else { fatalError("Diagnostic requires image input") }
        var usage:[String:Int] = [:], operations:[[String:String]] = []
        var planError = ""
        do {
            let plan = try await MLComputePlan.load(contentsOf:compiled,configuration:configuration)
            func walk(_ block:MLModelStructure.Program.Block) {
                for op in block.operations {
                    let name = plan.deviceUsage(for:op).map { usage -> String in
                        switch usage.preferred {
                        case .cpu: return "CPU"
                        case .gpu: return "GPU"
                        case .neuralEngine: return "ANE"
                        @unknown default: return "unknown"
                        }
                    } ?? "unreported"
                    usage[name,default:0] += 1
                    operations.append(["operation":op.operatorName,"preferredDevice":name])
                    op.blocks.forEach(walk)
                }
            }
            if case let .program(program) = plan.modelStructure {
                for (_,function) in program.functions { walk(function.block) }
            }
        } catch { planError = String(describing:error) }
        var runs:[[String:Any]] = []
        for level:UInt8 in (imagePath == nil ? [0,64,128,192,255] : [0]) {
            var buffer:CVPixelBuffer?
            let attrs:[CFString:Any] = [kCVPixelBufferIOSurfacePropertiesKey:[:],kCVPixelBufferMetalCompatibilityKey:true]
            guard CVPixelBufferCreate(nil,constraint.pixelsWide,constraint.pixelsHigh,kCVPixelFormatType_32BGRA,attrs as CFDictionary,&buffer) == kCVReturnSuccess,
                  let buffer else { fatalError("Cannot create probe image") }
            CVPixelBufferLockBaseAddress(buffer,[])
            let stride = CVPixelBufferGetBytesPerRow(buffer)
            let bytes = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to:UInt8.self)
            for y in 0..<constraint.pixelsHigh { for x in 0..<constraint.pixelsWide {
                let i = y*stride+x*4
                bytes[i] = level; bytes[i+1] = level; bytes[i+2] = level; bytes[i+3] = 255
            } }
            CVPixelBufferUnlockBaseAddress(buffer,[])
            if let imagePath {
                guard let image = CIImage(contentsOf:URL(fileURLWithPath:imagePath)) else { fatalError("Cannot load input image") }
                guard abs(image.extent.width/image.extent.height-Double(constraint.pixelsWide)/Double(constraint.pixelsHigh))<0.001 else { fatalError("Input aspect ratio mismatch; no center crop allowed") }
                let resized = image.transformed(by:CGAffineTransform(scaleX:Double(constraint.pixelsWide)/image.extent.width,y:Double(constraint.pixelsHigh)/image.extent.height))
                CIContext().render(resized,to:buffer,bounds:CGRect(x:0,y:0,width:constraint.pixelsWide,height:constraint.pixelsHigh),colorSpace:CGColorSpace(name:CGColorSpace.sRGB))
            }
            let provider = try MLDictionaryFeatureProvider(dictionary:[input.key:MLFeatureValue(pixelBuffer:buffer)])
            var times:[Double] = [], output:MLFeatureProvider?
            for iteration in 0..<102 {
                let start = ProcessInfo.processInfo.systemUptime
                output = try await model.prediction(from:provider)
                if iteration>=2 { times.append((ProcessInfo.processInfo.systemUptime-start)*1000) }
            }
            var record:[String:Any] = ["inputGrayByte":Int(level),"medianPredictionMS":times.sorted()[times.count/2]]
            if let imagePath { record.removeValue(forKey:"inputGrayByte"); record["inputImage"] = imagePath }
            let sorted = times.sorted()
            record["p95PredictionMS"] = sorted[Int(Double(sorted.count-1)*0.95)]
            record["warmSamples"] = times.count
            for name in output!.featureNames {
                guard let feature = output!.featureValue(for:name) else { continue }
                record["outputName"] = name; record["outputType"] = String(describing:feature.type)
                if let array = feature.multiArrayValue {
                    let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
                    record["outputShape"] = shape; record["outputStrides"] = strides
                    var low = Double.infinity, high = -Double.infinity, sum = 0.0
                    for linear in 0..<array.count {
                        var rest = linear, offset = 0
                        for axis in shape.indices.reversed() { offset += (rest%shape[axis])*strides[axis]; rest /= shape[axis] }
                        let value:Double
                        switch array.dataType {
                        case .float16: value = Double(array.dataPointer.assumingMemoryBound(to:Float16.self)[offset])
                        case .float32: value = Double(array.dataPointer.assumingMemoryBound(to:Float.self)[offset])
                        case .double: value = array.dataPointer.assumingMemoryBound(to:Double.self)[offset]
                        default: fatalError("Unsupported diagnostic output dtype")
                        }
                        low = min(low,value); high = max(high,value); sum += value
                    }
                    record["outputMin"] = low; record["outputMax"] = high; record["outputMean"] = sum/Double(array.count)
                    continue
                }
                guard let pixels = feature.imageBufferValue else { continue }
                record["outputPixelFormat"] = CVPixelBufferGetPixelFormatType(pixels)
                guard CVPixelBufferGetPixelFormatType(pixels) == kCVPixelFormatType_32BGRA else { continue }
                CVPixelBufferLockBaseAddress(pixels,.readOnly)
                let src = CVPixelBufferGetBaseAddress(pixels)!.assumingMemoryBound(to:UInt8.self)
                let row = CVPixelBufferGetBytesPerRow(pixels)
                var low = 255, high = 0, total:UInt64 = 0
                for y in 0..<CVPixelBufferGetHeight(pixels) { for x in 0..<CVPixelBufferGetWidth(pixels) { for c in 0..<3 {
                    let v = Int(src[y*row+x*4+c]); low = min(low,v); high = max(high,v); total += UInt64(v)
                } } }
                record["outputMinRGBByte"] = low; record["outputMaxRGBByte"] = high
                record["outputMeanRGBByte"] = Double(total)/Double(CVPixelBufferGetWidth(pixels)*CVPixelBufferGetHeight(pixels)*3)
                CVPixelBufferUnlockBaseAddress(pixels,.readOnly)
            }
            runs.append(record)
        }
        let report:[String:Any] = ["model":source.path,"configuration":"cpuAndNeuralEngine",
            "computePlanPreferredOperationCounts":usage,"computePlanOperations":operations,
            "computePlanError":planError,"probes":runs,
            "placementNote":"Compute-plan preference is not an Instruments trace of actual ANE utilization; inference timings exclude capture and GPU preparation."]
        print(String(data:try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
    }
}
