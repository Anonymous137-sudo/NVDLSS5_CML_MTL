import Foundation
import MLX

/// Small learned local affine residual, intended to be distilled from matched
/// captured-input / desired-output pairs. No default/random look is published.
/// It cannot synthesize new geometry, physically correct light transport or shadows.
public final class CompactAppearance {
    private let weights: [String:MLXArray]
    public init(url: URL) throws {
        let a = try loadArrays(url:url)
        let shapes = ["conv0.weight":[16,3,3,3],"conv0.bias":[16],
                      "conv1.weight":[16,3,3,16],"conv1.bias":[16],
                      "head.weight":[12,1,1,16],"head.bias":[12]]
        guard shapes.allSatisfy({ a[$0.key]?.shape == $0.value }) else {
            throw NSError(domain:"NeuralPost",code:4,userInfo:[NSLocalizedDescriptionKey:"Expected compact affine appearance weights"])
        }
        weights = a.mapValues { $0.asType(.float16) }; eval(Array(weights.values))
    }
    public func callAsFunction(_ rgb: MLXArray) -> MLXArray {
        // Pixel averages preserve coverage and avoid introducing another CPU bridge.
        var x = mean(rgb.reshaped(1,180,4,320,4,3).asType(.float32),axes:[2,4]).asType(.float16)
        x = maximum(conv2d(x,weights["conv0.weight"]!,stride:2,padding:1)+weights["conv0.bias"]!,0)
        x = maximum(conv2d(x,weights["conv1.weight"]!,stride:2,padding:1)+weights["conv1.bias"]!,0)
        return conv2d(x,weights["head.weight"]!)+weights["head.bias"]!
    }
}
