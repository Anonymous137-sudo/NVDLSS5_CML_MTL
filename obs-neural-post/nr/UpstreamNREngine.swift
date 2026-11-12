import CoreVideo
import DLSSMLX
import Foundation
import Metal
import MetalPerformanceShaders
import ObsDLSS5Native

/// Source and effect-filter entry points use the same recovered model contract.
/// The source adapter adds only sizing and the existing ProcessedFrame envelope.
final class UpstreamNREngine {
  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let scaler: MPSImageLanczosScale
  private let session: UnsafeMutableRawPointer
  private var controls: NRConfiguration
  private var reset = true

  init(device: MTLDevice, configuration: NRConfiguration) throws {
    self.device = device
    self.controls = configuration
    self.scaler = MPSImageLanczosScale(device: device)
    let resources = URL(fileURLWithPath: configuration.model71).deletingLastPathComponent().path
    guard let queue = device.makeCommandQueue(),
      let session = resources.withCString({ nr5Create($0) })
    else {
      throw MLXMediaError("Cannot create upstream NR source session")
    }
    self.queue = queue
    self.session = session
  }

  deinit { nr5Destroy(session) }

  func configure(_ settings: NRConfiguration) {
    if settings != controls {
      controls = settings
      reset = true
    }
  }

  func resetHistory() { reset = true }

  private func surface(width: Int, height: Int) throws -> (CVPixelBuffer, MTLTexture) {
    var pixels: CVPixelBuffer?
    let attrs: [CFString: Any] = [
      kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true,
    ]
    guard
      CVPixelBufferCreate(
        nil, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixels)
        == kCVReturnSuccess,
      let pixels, let io = CVPixelBufferGetIOSurface(pixels)?.takeUnretainedValue()
    else {
      throw MLXMediaError("Cannot allocate upstream NR surface")
    }
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
    descriptor.storageMode = .shared
    descriptor.usage = [.shaderRead, .shaderWrite]
    guard let texture = device.makeTexture(descriptor: descriptor, iosurface: io, plane: 0) else {
      throw MLXMediaError("Cannot create upstream NR texture")
    }
    return (pixels, texture)
  }

  private func texture(_ pixels: CVPixelBuffer) throws -> MTLTexture {
    guard CVPixelBufferGetPixelFormatType(pixels) == kCVPixelFormatType_32BGRA,
      let io = CVPixelBufferGetIOSurface(pixels)?.takeUnretainedValue()
    else {
      throw MLXMediaError("Upstream NR expects an owned BGRA IOSurface")
    }
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .bgra8Unorm,
      width: CVPixelBufferGetWidth(pixels), height: CVPixelBufferGetHeight(pixels), mipmapped: false
    )
    descriptor.storageMode = .shared
    descriptor.usage = .shaderRead
    guard let texture = device.makeTexture(descriptor: descriptor, iosurface: io, plane: 0) else {
      throw MLXMediaError("Cannot import upstream NR source")
    }
    return texture
  }

  private func scale(_ source: MTLTexture, to destination: MTLTexture) throws {
    guard let command = queue.makeCommandBuffer() else {
      throw MLXMediaError("Cannot submit NR resize")
    }
    scaler.encode(commandBuffer: command, sourceTexture: source, destinationTexture: destination)
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw command.error ?? MLXMediaError("NR resize failed")
    }
  }

  func process(
    _ input: CVPixelBuffer, _ sequence: UInt64, _ pts: Double,
    captureCorrection: Bool = false
  ) throws -> ProcessedFrame {
    let start = ProcessInfo.processInfo.systemUptime
    let width = CVPixelBufferGetWidth(input)
    let height = CVPixelBufferGetHeight(input)
    let budget = controls.nativeProcessingHeight ?? 1080
    let ratio =
      budget > 0
      ? min(
        1,
        min(
          Double(budget) / Double(height),
          Double(budget * 16 / 9) / Double(width))) : 1
    let w = max(2, Int(Double(width) * ratio) / 2 * 2)
    let h = max(2, Int(Double(height) * ratio) / 2 * 2)
    let work: CVPixelBuffer
    if w != width || h != height {
      let resized = try surface(width: w, height: h)
      try scale(texture(input), to: resized.1)
      work = resized.0
    } else {
      work = input
    }
    let mode = ["71-480": 0, "31-480": 1, "31then71-480": 2, "71x2-480": 3, "31x2-480": 4][
      controls.mode]!
    let options: [String: Any] = [
      "mode": mode, "style": controls.style, "preset": 0,
      "intensity": controls.layerGain, "finalIntensity": controls.finalGain,
      "tone": controls.tone, "structure": controls.structure,
      "skin": controls.skinStructure, "maskStructure": controls.maskStructure,
      "automask": controls.automaticMask, "temporal": controls.temporalNR == true,
      "reset": reset, "channelOrder": 0, "model71": controls.model71, "model31": controls.model31,
    ]
    let settings = String(
      decoding: try JSONSerialization.data(withJSONObject: options), as: UTF8.self)
    let output = settings.withCString { text in
      nr5ProcessSurface(session, Unmanaged.passUnretained(work).toOpaque(), text,
                        UInt64(max(0, pts) * 1_000_000_000))
    }
    guard let output else {
      var message = [CChar](repeating: 0, count: 1024)
      nr5Error(session, &message, message.count)
      throw MLXMediaError(String(cString: message))
    }
    let native = Unmanaged<CVPixelBuffer>.fromOpaque(output).takeRetainedValue()
    reset = false
    let result: (CVPixelBuffer, MTLTexture)
    if w != width || h != height {
      result = try surface(width: width, height: height)
      try scale(texture(native), to: result.1)
    } else {
      result = (native, try texture(native))
    }
    let elapsed = (ProcessInfo.processInfo.systemUptime - start) * 1000
    let frame = ProcessedFrame(result.0, result.1, sequence, pts, elapsed, 0)
    frame.stages = [
      "upstreamNativeNR": 1, "nativeNRPasses": mode > 1 ? 2 : 1,
      "nativeNRWidth": Double(w), "nativeNRHeight": Double(h), "nativeNRFP16": 1, "nativeSurfaceBridge": 1,
    ]
    var profile = [Double](repeating: 0, count: 8)
    if nr5Profile(session, &profile, profile.count) == profile.count {
      let names = ["nativeImportMS", "nativeFlowMS", "nativePass1MS", "nativePass2MS",
                   "nativeExportMS", "nativeHistoryReset", "nativeInputCopyMS", "nativeOutputCopyMS"]
      for (name, value) in zip(names, profile) { frame.stages[name] = value }
    }
    _ = captureCorrection
    return frame
  }
}
