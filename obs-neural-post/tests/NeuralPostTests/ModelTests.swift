import Foundation
import MLX
import XCTest
@testable import NeuralPost

final class ModelTests: XCTestCase {
    func testConvertedModelAgainstUnfusedPyTorchReference() throws {
        guard let path = ProcessInfo.processInfo.environment["NEURAL_POST_WEIGHTS"] else {
            throw XCTSkip("Set NEURAL_POST_WEIGHTS to run real checkpoint parity")
        }
        let url = URL(fileURLWithPath: path)
        let fixture = try loadArrays(url: url.deletingPathExtension().appendingPathExtension("parity.safetensors"))
        for half in [false,true] {
            let model = try CompactSR(url: url,half: half)
            let output = model(fixture["input"]!.asType(half ? .float16 : .float32)).asType(.float32)
            let delta = abs(output-fixture["expected"]!)
            let maximum = max(delta).item(Float.self), average = mean(delta).item(Float.self)
            print("PARITY half=\(half) max=\(maximum) mean=\(average)")
            XCTAssertEqual(output.shape,fixture["expected"]!.shape)
            XCTAssertLessThan(maximum,half ? 0.03 : 0.0002)
            XCTAssertLessThan(average,half ? 0.003 : 0.00002)
        }
    }
    func testRejectsInvalidModelArchive() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays:["bad":MLXArray([Float(0)])],url:url)
        XCTAssertThrowsError(try CompactSR(url:url))
    }
}
