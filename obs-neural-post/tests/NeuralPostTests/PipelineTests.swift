import CoreVideo
import Foundation
import MLX
import XCTest
@testable import NeuralPost

final class PipelineTests: XCTestCase {
    private func input() throws -> CVPixelBuffer {
        var pixels:CVPixelBuffer?
        let attrs:[CFString:Any] = [kCVPixelBufferIOSurfacePropertiesKey:[:],kCVPixelBufferMetalCompatibilityKey:true]
        XCTAssertEqual(CVPixelBufferCreate(nil,1280,720,kCVPixelFormatType_32BGRA,attrs as CFDictionary,&pixels),kCVReturnSuccess)
        let image = try XCTUnwrap(pixels)
        CVPixelBufferLockBaseAddress(image,[])
        let base = CVPixelBufferGetBaseAddress(image)!.assumingMemoryBound(to:UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(image)
        for y in 0..<720 { for x in 0..<1280 {
            let i = y*stride+x*4
            base[i] = UInt8(x%256); base[i+1] = UInt8(y%256)
            base[i+2] = UInt8((x+y)%256); base[i+3] = 255
        } }
        CVPixelBufferUnlockBaseAddress(image,[])
        return image
    }
    func testFullFrameNeuralPackingAgainstUnfusedReference() throws {
        guard let path = ProcessInfo.processInfo.environment["NEURAL_POST_WEIGHTS"] else { throw XCTSkip("Set NEURAL_POST_WEIGHTS") }
        let url = URL(fileURLWithPath:path)
        let fixture = url.deletingLastPathComponent().appendingPathComponent("full-parity.json")
        let samples = try JSONSerialization.jsonObject(with:Data(contentsOf:fixture)) as! [[String:Any]]
        let pipeline = try PostPipeline(modelURL:url)
        defer { pipeline.stop() }
        let completed = expectation(description:"full neural frame")
        var output:ProcessedFrame?
        pipeline.onComplete = { output = $0; completed.fulfill() }
        pipeline.submit(try input(),sequence:7,sourceSeconds:ProcessInfo.processInfo.systemUptime)
        wait(for:[completed],timeout:30)
        let frame = try XCTUnwrap(output)
        XCTAssertTrue(frame.neural); XCTAssertEqual(frame.sequence,7)
        XCTAssertEqual(frame.texture.width,3840); XCTAssertEqual(frame.texture.height,2160)
        CVPixelBufferLockBaseAddress(frame.pixels,.readOnly)
        defer { CVPixelBufferUnlockBaseAddress(frame.pixels,.readOnly) }
        let base = CVPixelBufferGetBaseAddress(frame.pixels)!.assumingMemoryBound(to:UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(frame.pixels)
        var maximum:Double = 0
        for point in samples {
            let x = point["x"] as! Int, y = point["y"] as! Int, rgb = point["rgb"] as! [Double]
            for c in 0..<3 { maximum = Swift.max(maximum,Swift.abs(Double(base[y*stride+x*4+2-c])/255-rgb[c])) }
            XCTAssertEqual(base[y*stride+x*4+3],255)
        }
        print("FULL-PIPELINE reference max sample error=\(maximum)")
        XCTAssertLessThan(maximum,0.035)
        XCTAssertEqual(pipeline.stats.failed,0)
    }
    func testCapturePressureAndRetiredGenerationDoNotQueueOrPublish() throws {
        let pipeline = try PostPipeline()
        let pixels = try input()
        for i in 1...240 { pipeline.submit(pixels,sequence:UInt64(i),sourceSeconds:0) }
        XCTAssertLessThanOrEqual(pipeline.stats.pending,1)
        XCTAssertGreaterThan(pipeline.stats.superseded,200)
        pipeline.reset(); pipeline.stop()
        for _ in 0..<200 where pipeline.stats.busy { Thread.sleep(forTimeInterval:0.01) }
        XCTAssertFalse(pipeline.stats.busy)
        XCTAssertEqual(pipeline.stats.pending,0)
        XCTAssertNil(pipeline.latest())
        let submitted = pipeline.stats.submitted
        pipeline.submit(pixels,sequence:999,sourceSeconds:0)
        XCTAssertEqual(pipeline.stats.submitted,submitted)
    }
    func testMissingModelDoesNotSilentlyBecomeSpatialOutput() throws {
        let pipeline = try PostPipeline(modelURL:URL(fileURLWithPath:"/nonexistent/neural-post.safetensors"))
        defer { pipeline.stop() }
        pipeline.submit(try input(),sequence:1,sourceSeconds:0)
        for _ in 0..<200 where pipeline.stats.busy { Thread.sleep(forTimeInterval:0.01) }
        XCTAssertEqual(pipeline.stats.completed,0)
        XCTAssertGreaterThan(pipeline.stats.failed,0)
        XCTAssertFalse(pipeline.stats.error.isEmpty)
    }
    func testAppearanceIdentityCoefficientsAndShape() throws {
        let shapes = ["conv0.weight":[16,3,3,3],"conv0.bias":[16],"conv1.weight":[16,3,3,16],"conv1.bias":[16],"head.weight":[12,1,1,16],"head.bias":[12]]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".safetensors")
        defer { try? FileManager.default.removeItem(at:url) }
        try save(arrays:shapes.mapValues { MLXArray.zeros($0) },url:url)
        let model = try CompactAppearance(url:url)
        let output = model(MLXArray.ones([1,720,1280,3],dtype:.float16))
        XCTAssertEqual(output.shape,[1,45,80,12])
        XCTAssertEqual(max(abs(output)).item(Float.self),0)
    }
}
