import CoreVideo
import Foundation
import Metal
import RifeMetal

final class RifeFrameInterpolator {
    private let device: MTLDevice
    private let model: RifeInterpolator
    private var stream: RifeStream?
    private var previous: (sequence: UInt64, seconds: Double, signature: [UInt8], appearance: ProcessedFrame?)?
    private var width = 0
    private var height = 0

    init(device: MTLDevice, modelURL: URL) throws {
        self.device = device
        model = try RifeInterpolator(configuration: .init(modelURL: modelURL, qualityTier: .fast, preferredDevice: device))
    }

    func reset() {
        stream?.reset()
        previous = nil
    }

    func generate(_ base: ProcessedFrame, raw: CVPixelBuffer?, factor: Int) throws -> MultiFrameInterpolator.Result {
        let start = ProcessInfo.processInfo.systemUptime
        let w = base.texture.width
        let h = base.texture.height
        if stream == nil || width != w || height != h {
            stream = try model.makeStream(width: w, height: h)
            width = w
            height = h
            previous = nil
        }
        let motionInput = raw ?? base.pixels
        let signature = try sample(motionInput)
        let earlier = previous
        let gap = earlier.map { base.sourceSeconds - $0.seconds } ?? 0
        let difference = earlier.map { before in
            Double(zip(before.signature, signature).reduce(0) { $0 + abs(Int($1.0) - Int($1.1)) })
                / Double(signature.count * 255)
        } ?? 0
        let cut = earlier == nil || gap <= 0 || gap > 0.25 || difference >= 0.25
        let phases = cut ? [] : (1..<factor).map { Float($0) / Float(factor) }
        let appearance = raw.flatMap { _ in earlier?.appearance.map { ($0.texture, base.texture) } }
        let buffers = try stream!.push(motionInput, timesteps: phases, appearance: appearance)
        guard buffers.count == phases.count else { throw NSError(domain: "RIFE Metal stream", code: 1) }
        previous = (base.sequence, base.sourceSeconds, signature, raw == nil ? nil : base)
        base.stages["rifeRawMotion"] = 1
        base.stages["rifeNeuralWarp"] = appearance == nil || phases.isEmpty ? 0 : 1

        let frames = try zip(buffers, phases).map { pixels, phase in
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
            descriptor.storageMode = .shared
            descriptor.usage = .shaderRead
            guard let surface = CVPixelBufferGetIOSurface(pixels)?.takeUnretainedValue(),
                  let texture = device.makeTexture(descriptor: descriptor, iosurface: surface, plane: 0) else {
                throw NSError(domain: "RIFE Metal output", code: 1)
            }
            let time = earlier!.seconds + Double(phase) * gap
            let frame = ProcessedFrame(pixels, texture, base.sequence, time, 0, 0)
            frame.generated = true
            frame.phase = Double(phase)
            frame.pairStart = earlier!.sequence
            frame.pairEnd = base.sequence
            return frame
        }
        return .init(frames: frames, milliseconds: (ProcessInfo.processInfo.systemUptime - start) * 1000,
                     cut: cut, difference: difference, validationMaxDelta: -1)
    }

    private func sample(_ pixels: CVPixelBuffer) throws -> [UInt8] {
        guard CVPixelBufferGetPixelFormatType(pixels) == kCVPixelFormatType_32BGRA,
              CVPixelBufferLockBaseAddress(pixels, .readOnly) == kCVReturnSuccess else {
            throw NSError(domain: "RIFE Metal input", code: 1)
        }
        defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
        guard let bytes = CVPixelBufferGetBaseAddress(pixels)?.assumingMemoryBound(to: UInt8.self) else {
            throw NSError(domain: "RIFE Metal input", code: 2)
        }
        let rowBytes = CVPixelBufferGetBytesPerRow(pixels)
        let width = CVPixelBufferGetWidth(pixels)
        let height = CVPixelBufferGetHeight(pixels)
        var signature: [UInt8] = []
        signature.reserveCapacity(32 * 18 * 3)
        for y in 0..<18 {
            for x in 0..<32 {
                let offset = ((2 * y + 1) * height / 36) * rowBytes + ((2 * x + 1) * width / 64) * 4
                signature.append(bytes[offset])
                signature.append(bytes[offset + 1])
                signature.append(bytes[offset + 2])
            }
        }
        return signature
    }
}
