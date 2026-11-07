import Foundation
import DLSSCore
import DLSSMLX
import MLX

@main struct NativeTeacherBudget {
    static func main() async throws {
        guard [4,5,7].contains(CommandLine.arguments.count) else { fatalError("MODEL FEATURES METALLIB") }
        // MLX 0.31 loads the colocated mlx.metallib copied by the build helper.
        let mode=CommandLine.arguments.count>=5 ? (MLXExecutionMode(rawValue:CommandLine.arguments[4]) ?? .metalFused) : .metalFused
        let width=CommandLine.arguments.count==7 ? Int(CommandLine.arguments[5])! : 1280
        let height=CommandLine.arguments.count==7 ? Int(CommandLine.arguments[6])! : 768
        let model=try MLXNeuralRenderer(packageURL:URL(fileURLWithPath:CommandLine.arguments[1]),executionMode:mode,computePrecision:.float16)
        let input=try HostTensor(descriptor:TensorDescriptor(name:"color",shape:[1,height,width,16],dataType:.float32,layout:.nhwc),bytes:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[2])))
        var wall:[Double]=[],execution:[Double]=[]
        for i in 0..<7 {
            let began=ProcessInfo.processInfo.systemUptime
            let output=try await model.render(NeuralRenderRequest(sequenceID:UInt64(i),inputs:[input]))
            let elapsed=(ProcessInfo.processInfo.systemUptime-began)*1000
            if i>0 { wall.append(elapsed);execution.append(Double(output.timing.executionNanoseconds)/1e6) }
        }
        let a=execution.sorted(),b=wall.sorted(),memory=MLXRuntimeDiagnostics.memorySnapshot()
        let report:[String:Any] = ["model":"full HT, one pass","networkWidth":width,"networkHeight":height,"executionMode":mode.rawValue, "warmSamples":a.count,
            "forwardExecutionMS":execution,"forwardMedianMS":a[a.count/2],"renderAndReadbackMS":wall,"renderMedianMS":b[b.count/2],
            "mlxPeakActiveBytes":memory.peakActiveBytes,"note":"Precomputed feature input; forward timer includes MLX eval and fp32 output conversion, excludes host readback and preparation. Not a Metal GPU timestamp or HDMI/SR/FG/OBS benchmark."]
        print(String(data:try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
    }
}
