import Foundation
import MLX

/// RT4KSR's deployed x3 branch: pixel-unshuffle → 24-channel conv/normalization
/// backbone → 6x pixel-shuffle. Training-only branches are fused by the converter.
/// RGB is display-referred SDR in [0, 1]. No recovered DLSS transformer executes.
public final class CompactSR {
    private let weights: [String: MLXArray]
    private lazy var compiled = compile { [unowned self] (input: MLXArray) in self.forward(input) }
    private static let normalize24 = MLXFast.metalKernel(name: "np_norm24", inputNames: ["x", "weight", "bias"], outputNames: ["out"], source: """
        uint lane=thread_position_in_grid.x, row=thread_position_in_grid.y;
        if(row>=ROWS) return;
        float v=lane<24 ? float(x[row*24+lane]) : 0.f;
        float mu=simd_sum(v)/24.f;
        float delta=lane<24 ? v-mu : 0.f;
        float inv=rsqrt(simd_sum(delta*delta)/24.f+1e-6f);
        if(lane<24) out[row*24+lane]=(v-mu)*inv*float(weight[lane])+float(bias[lane]);
        """)
    private static let activateNormalize24 = MLXFast.metalKernel(name: "np_activate_norm24", inputNames: ["x", "input_bias", "weight", "bias"], outputNames: ["out"], source: """
        uint lane=thread_position_in_grid.x, row=thread_position_in_grid.y;
        if(row>=ROWS) return;
        float v=lane<24 ? float(half(x[row*24+lane])+half(input_bias[lane])) : 0.f;
        if(ACTIVATE && lane<24) {
            float argument=clamp(.7978845608f*(v+.044715f*v*v*v),-9.f,9.f);
            v=float(half(.5f*v*(1.f+tanh(argument))));
        }
        float mu=simd_sum(v)/24.f;
        float delta=lane<24 ? v-mu : 0.f;
        float inv=rsqrt(simd_sum(delta*delta)/24.f+1e-6f);
        if(lane<24) out[row*24+lane]=(v-mu)*inv*float(weight[lane])+float(bias[lane]);
        """)
    public init(url: URL, half: Bool = true) throws {
        let loaded = try loadArrays(url: url)
        let names = ["head.0.weight", "head.0.bias", "tail.0.weight", "tail.0.bias",
                     "tail.1.rep_conv.weight", "tail.1.rep_conv.bias", "upsample.0.weight", "upsample.0.bias"]
            + (0..<4).flatMap { ["body.\($0).norm.weight", "body.\($0).norm.bias",
                                "body.\($0).conv1.rep_conv.weight", "body.\($0).conv1.rep_conv.bias"] }
        func expected(_ name:String) -> [Int] {
            if name == "head.0.weight" { return [24,3,3,12] }
            if name == "upsample.0.weight" { return [108,3,3,24] }
            if name == "upsample.0.bias" { return [108] }
            if name.contains("norm.") || name.hasPrefix("tail.0.") || name.hasSuffix(".bias") { return [24] }
            return [24,3,3,24]
        }
        guard names.allSatisfy({ name in
            guard let value = loaded[name] else { return false }
            return value.shape == expected(name) && (value.dtype == .float32 || value.dtype == .float16)
        }) else {
            throw NSError(domain: "NeuralPost", code: 1, userInfo: [NSLocalizedDescriptionKey: "Expected converted RT4KSR x3 weights (24 features, four blocks)"])
        }
        weights = Dictionary(uniqueKeysWithValues:names.map { ($0,loaded[$0]!.asType(half ? .float16 : .float32)) })
        eval(Array(weights.values))
        guard weights.values.allSatisfy({ all(isFinite($0)).item(Bool.self) }) else {
            throw NSError(domain:"NeuralPost",code:1,userInfo:[NSLocalizedDescriptionKey:"Non-finite SR model weights"])
        }
    }
    private func conv(_ x: MLXArray, _ name: String) -> MLXArray {
        conv2d(x, weights[name + ".weight"]!, padding: 1) + weights[name + ".bias"]!
    }
    private func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        if ProcessInfo.processInfo.environment["NEURAL_POST_NORM"] == "reference" {
            let f = x.asType(.float32), mu = mean(x.asType(.float32), axis: -1, keepDims: true)
            let centered = f-mu
            return ((centered*rsqrt(mean(centered*centered,axis:-1,keepDims:true)+1e-6))*weights[name+".weight"]!+weights[name+".bias"]!).asType(x.dtype)
        }
        let rows = x.size/24
        return Self.normalize24([x,weights[name+".weight"]!,weights[name+".bias"]!],
            template:[("ROWS",rows)],grid:(32,rows,1),threadGroup:(32,8,1),
            outputShapes:[x.shape],outputDTypes:[x.dtype])[0]
    }
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let x = head(input) + weights["upsample.0.bias"]!
        let h = input.shape[1], w = input.shape[2]
        return x.reshaped(1, h/2, w/2, 3, 6, 6)
            .transposed(0, 1, 4, 2, 5, 3).reshaped(1, h*3, w*3, 3)
    }
    /// Leave subpixels in LR channel order. Metal fuses shuffle, clipping and
    /// BGRA packing, avoiding a full 4K RGB transpose/materialization.
    public func head(_ input: MLXArray) -> MLXArray {
        if input.dtype == .float16 && ProcessInfo.processInfo.environment["NEURAL_POST_UNFUSED"] != "1" { return fusedForward(input) }
        return ProcessInfo.processInfo.environment["NEURAL_POST_COMPILE"] == "1" ? compiled(input) : forward(input)
    }
    public var outputBias: MLXArray { weights["upsample.0.bias"]! }
    private func fusedNorm(_ x: MLXArray, inputBias: MLXArray, name: String, activate: Bool) -> MLXArray {
        let rows = x.size/24
        return Self.activateNormalize24([x,inputBias,weights[name+".weight"]!,weights[name+".bias"]!],
            template:[("ROWS",rows),("ACTIVATE",activate)],grid:(32,rows,1),threadGroup:(32,8,1),
            outputShapes:[x.shape],outputDTypes:[x.dtype])[0]
    }
    private func fusedForward(_ input: MLXArray) -> MLXArray {
        let h = input.shape[1], w = input.shape[2]
        let low = input.reshaped(1,h/2,2,w/2,2,3).transposed(0,1,3,5,2,4).reshaped(1,h/2,w/2,12)
        var x = conv2d(low,weights["head.0.weight"]!,padding:1)
        var inputBias = weights["head.0.bias"]!
        for i in 0..<4 {
            x = fusedNorm(x,inputBias:inputBias,name:"body.\(i).norm",activate:i>0)
            x = conv2d(x,weights["body.\(i).conv1.rep_conv.weight"]!,padding:1)
            inputBias = weights["body.\(i).conv1.rep_conv.bias"]!
        }
        x = fusedNorm(x,inputBias:inputBias,name:"tail.0",activate:true)
        x = conv(x,"tail.1.rep_conv")
        return conv2d(x,weights["upsample.0.weight"]!,padding:1)
    }
    private func forward(_ input: MLXArray) -> MLXArray {
        let h = input.shape[1], w = input.shape[2]
        precondition(input.shape == [1, h, w, 3] && h % 2 == 0 && w % 2 == 0)
        // PyTorch pixel-unshuffle order is channel, row-offset, column-offset.
        let low = input.reshaped(1, h/2, 2, w/2, 2, 3)
            .transposed(0, 1, 3, 5, 2, 4).reshaped(1, h/2, w/2, 12)
        var x = conv(low, "head.0")
        for i in 0..<4 {
            x = conv(norm(x, "body.\(i).norm"), "body.\(i).conv1.rep_conv")
            x = x * 0.5 * (1 + erf(x / Float(sqrt(2.0))))
        }
        x = conv(norm(x, "tail.0"), "tail.1.rep_conv")
        x = conv2d(x, weights["upsample.0.weight"]!, padding: 1)
        return x
    }
}
