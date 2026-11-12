// Native implementation of the obs-dlss5-nr bridge. GPL-2.0-or-later.
import CoreVideo
import DLSSCore
import DLSSMLX
import DLSSMedia
import Foundation
import Metal

private struct Options: Decodable, Equatable {
  let style: Int
  let preset: Int
  let intensity: Float
  let tone: Float
  let structure: Float
  let skin: Float
  let automask: Bool
  let temporal: Bool
  let reset: Bool
  let mode: Int
  let channelOrder: Int
  let model71: String?
  let model31: String?
  let finalIntensity: Float?
  let maskStructure: Float?

  func validate() throws {
    guard (0...128).contains(style), preset == 0, (0...4).contains(mode),
      (0...2).contains(intensity), (0...2).contains(tone),
      (0...2).contains(structure), (-1...2).contains(skin),
      (0...2).contains(channelOrder), (0...2).contains(finalIntensity ?? 1),
      (-1...2).contains(maskStructure ?? -1)
    else {
      throw MLXMediaError("Invalid NR controls or unsupported NVIDIA render preset")
    }
  }
}

private struct NativeResult: Sendable {
  let pixels: MLXPixelBuffer
  // Import, optical flow, pass 1, pass 2, export, history reset.
  let profile: [Double]
}

private actor NativeSession {
  private let resources: URL
  private var backends: [MLXNeuralRenderingDeviceTemporalBackend] = []
  private var modelSelection: String?
  private var previousStyle: Int?
  private var previousTemporal: Bool?
  private var opticalFlow: NativeOpticalFlow?
  private var previousTime: UInt64?
  private var frameIndex: UInt64 = 0
  private var extent = [Int]()
  private var creationFailure: (key: String, time: Double, message: String)?
  private var writer: MLXPixelBufferWriter?

  init(resources: URL) { self.resources = resources }

  private func clear() async {
    for backend in backends { await backend.reset(sequenceID: nil) }
    await opticalFlow?.reset()
    previousTime = nil
    frameIndex = 0
  }

  func process(_ pixels: MLXPixelBuffer, options: Options, timestamp: UInt64, reset: Bool)
    async throws -> NativeResult
  {
    try options.validate()
    let size = [pixels.width, pixels.height]
    let selection =
      "\(options.mode)/\(options.model71 ?? resources.path)/\(options.model31 ?? resources.path)"
    let creationKey = "\(selection)/\(options.style)/\(options.preset)/\(options.temporal)/\(size)"
    if reset || options.reset { creationFailure = nil }
    if let failure = creationFailure, failure.key == creationKey,
      ProcessInfo.processInfo.systemUptime - failure.time < 5
    {
      throw MLXMediaError(failure.message)
    }
    if modelSelection != selection {
      do {
        await clear()
        backends = []
        modelSelection = nil
        let names: [String]
        switch options.mode {
        case 1: names = ["NeuralRendering-Small"]
        case 2: names = ["NeuralRendering-Small", "NeuralRendering"]
        case 3: names = ["NeuralRendering", "NeuralRendering"]
        case 4: names = ["NeuralRendering-Small", "NeuralRendering-Small"]
        default: names = ["NeuralRendering"]
        }
        for name in names {
          let override = name == "NeuralRendering" ? options.model71 : options.model31
          let url =
            override.map { URL(fileURLWithPath: $0) }
            ?? resources.appendingPathComponent(name + ".dlssmodel")
          guard FileManager.default.fileExists(atPath: url.path) else {
            throw MLXMediaError("Missing model package at \(url.path)")
          }
          backends.append(
            try MLXNeuralRenderingDeviceTemporalBackend(
              packageURL: url, executionMode: .metalFused, computePrecision: .float16, depthInverted: true))
        }
        modelSelection = selection
        creationFailure = nil
      } catch {
        creationFailure = (
          creationKey, ProcessInfo.processInfo.systemUptime, String(describing: error)
        )
        throw error
      }
    }
    let timeDiscontinuity =
      previousTime.map { timestamp <= $0 || timestamp - $0 > 250_000_000 } ?? false
    let historyReset = reset || options.reset || size != extent || previousStyle != options.style
      || previousTemporal != options.temporal || timeDiscontinuity
    if historyReset {
      await clear()
    }
    if size != extent {
      writer = try MLXPixelBufferWriter(width: pixels.width, height: pixels.height)
      opticalFlow = nil
      extent = size
    }
    var profile = [Double](repeating: 0, count: 6)
    profile[5] = historyReset ? 1 : 0
    var stageStart = ProcessInfo.processInfo.systemUptime
    let input = try MLXVideoFrame(pixelBuffer: pixels)
    profile[0] = (ProcessInfo.processInfo.systemUptime - stageStart) * 1000
    stageStart = ProcessInfo.processInfo.systemUptime
    var motion: MLXVideoMotion?
    if options.temporal {
      if opticalFlow == nil {
        opticalFlow = try NativeOpticalFlow(width: pixels.width, height: pixels.height)
      }
      motion = try await opticalFlow!.prepare(input, index: Int(frameIndex), sceneCutThreshold: 0.3)
    }
    profile[1] = (ProcessInfo.processInfo.systemUptime - stageStart) * 1000
    let controls = NeuralRenderingFeatureControls(
      normalizedStyle: Float(options.style) / 128,
      localToneStrength: options.tone, localStructureStrength: options.structure,
      automaticMask: options.automask
        ? .init(
          skinStructureStrength: options.skin,
          automaticMaskStructureStrength: options.maskStructure ?? -1) : nil)
    let context = NeuralRenderFrameContext(streamID: 1, frameIndex: frameIndex)
    let outputOptions = try MLXVideoOutputOptions(width: pixels.width, height: pixels.height)
    do {
      var image = input
      for (index, backend) in backends.enumerated() {
        stageStart = ProcessInfo.processInfo.systemUptime
        image = try await backend.renderVideoFrame(
          image, motion: motion, context: context,
          temporal: options.temporal, outputOptions: outputOptions,
          featureControls: controls, intensity: options.intensity,
          diagnosticLabel: "capture-\(timestamp)-pass-\(index + 1)")
        // renderVideoFrame evaluates before returning: these include completed GPU work.
        profile[2 + index] = (ProcessInfo.processInfo.systemUptime - stageStart) * 1000
      }
      stageStart = ProcessInfo.processInfo.systemUptime
      image = try image.applyingIntensity(over: input, strength: options.finalIntensity ?? 1)
      guard let writer else { throw MLXMediaError("Output writer unavailable") }
      let result = try await writer.write(image)
      previousTime = timestamp
      previousStyle = options.style
      previousTemporal = options.temporal
      frameIndex &+= 1
      profile[4] = (ProcessInfo.processInfo.systemUptime - stageStart) * 1000
      return NativeResult(pixels: result, profile: profile)
    } catch {
      await clear()
      throw error
    }
  }
}

private func makeSurface(width: Int, height: Int, format: OSType = kCVPixelFormatType_32BGRA) throws
  -> CVPixelBuffer
{
  var pixels: CVPixelBuffer?
  let attributes: [CFString: Any] = [
    kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true,
  ]
  let status = CVPixelBufferCreate(nil, width, height, format, attributes as CFDictionary, &pixels)
  guard status == kCVReturnSuccess, let pixels else {
    throw MLXMediaError("Cannot allocate NR IOSurface (\(status))")
  }
  return pixels
}

// The C++ worker owns one context and serializes calls. The actor owns MLX
// arrays and history. The semaphore waits only on that caller's worker thread.
private final class NativeContext: @unchecked Sendable {
  let lock = NSLock()
  let session: NativeSession
  var error = ""
  var profile = [Double](repeating: 0, count: 8)
  var reset = false
  init(resources: URL) { session = NativeSession(resources: resources) }

  // The caller holds lock and owns the input until this worker wait finishes.
  func render(_ pixels: CVPixelBuffer, options: Options, timestamp: UInt64) throws -> NativeResult {
    let completion = Completion()
    let session = session
    let reset = reset
    self.reset = false
    Task.detached {
      do {
        completion.result = .success(
          try await session.process(MLXPixelBuffer(pixels), options: options,
                                    timestamp: timestamp, reset: reset))
      } catch { completion.result = .failure(error) }
      completion.semaphore.signal()
    }
    completion.semaphore.wait()
    return try completion.result!.get()
  }
}

private final class Completion: @unchecked Sendable {
  let semaphore = DispatchSemaphore(value: 0)
  var result: Result<NativeResult, Error>?
}

@_cdecl("nr5_create")
public func nr5Create(_ path: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
  guard let path, MTLCreateSystemDefaultDevice() != nil else { return nil }
  return Unmanaged.passRetained(
    NativeContext(resources: URL(fileURLWithPath: String(cString: path)))
  ).toOpaque()
}

@_cdecl("nr5_destroy")
public func nr5Destroy(_ handle: UnsafeMutableRawPointer?) {
  guard let handle else { return }
  _ = Unmanaged<NativeContext>.fromOpaque(handle).takeRetainedValue()
}

@_cdecl("nr5_reset")
public func nr5Reset(_ handle: UnsafeMutableRawPointer?) {
  guard let handle else { return }
  let context = Unmanaged<NativeContext>.fromOpaque(handle).takeUnretainedValue()
  context.lock.lock()
  context.reset = true
  context.lock.unlock()
}

@_cdecl("nr5_error")
public func nr5Error(
  _ handle: UnsafeMutableRawPointer?, _ output: UnsafeMutablePointer<CChar>?, _ capacity: Int
) {
  guard let handle, let output, capacity > 0 else { return }
  let context = Unmanaged<NativeContext>.fromOpaque(handle).takeUnretainedValue()
  context.lock.lock()
  defer { context.lock.unlock() }
  context.error.withCString { text in
    let count = min(capacity - 1, strlen(text))
    memcpy(output, text, count)
    output[count] = 0
  }
}

// The worker reads this after nr5_process. Values are completed wall times in ms,
// except index 5 (history reset flag); indices 6/7 are C input/output copies.
@_cdecl("nr5_profile")
public func nr5Profile(_ handle: UnsafeMutableRawPointer?, _ output: UnsafeMutablePointer<Double>?, _ capacity: Int) -> Int {
  guard let handle, let output, capacity > 0 else { return 0 }
  let context = Unmanaged<NativeContext>.fromOpaque(handle).takeUnretainedValue()
  context.lock.lock()
  defer { context.lock.unlock() }
  let count = min(capacity, context.profile.count)
  for index in 0..<count { output[index] = context.profile[index] }
  return count
}

@_cdecl("nr5_process")
public func nr5Process(
  _ handle: UnsafeMutableRawPointer?, _ input: UnsafePointer<UInt8>?, _ inputPitch: Int32,
  _ output: UnsafeMutablePointer<UInt8>?, _ outputPitch: Int32, _ width: Int32, _ height: Int32,
  _ settings: UnsafePointer<CChar>?, _ timestamp: UInt64
) -> Bool {
  guard let handle else { return false }
  let context = Unmanaged<NativeContext>.fromOpaque(handle).takeUnretainedValue()
  context.lock.lock()
  defer { context.lock.unlock() }
  do {
    guard let input, let output, let settings, width > 0, height > 0, width <= 8192, height <= 8192,
      inputPitch >= width * 4, outputPitch >= width * 4
    else {
      throw MLXMediaError("Invalid BGRA dimensions, pointers or row pitches")
    }
    let options = try JSONDecoder().decode(Options.self, from: Data(String(cString: settings).utf8))
    try options.validate()
    let copyStart = ProcessInfo.processInfo.systemUptime
    let pixels = try makeSurface(width: Int(width), height: Int(height))
    guard CVPixelBufferLockBaseAddress(pixels, []) == kCVReturnSuccess else {
      throw MLXMediaError("Cannot map NR input")
    }
    let base = CVPixelBufferGetBaseAddress(pixels)!
    let stride = CVPixelBufferGetBytesPerRow(pixels)
    for y in 0..<Int(height) {
      let source = input.advanced(by: y * Int(inputPitch))
      let destination = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
      memcpy(destination, source, Int(width) * 4)
      if options.channelOrder == 1 {
        for x in 0..<Int(width) {
          let red = destination[x * 4]
          destination[x * 4] = destination[x * 4 + 2]
          destination[x * 4 + 2] = red
        }
      }
    }
    CVPixelBufferUnlockBaseAddress(pixels, [])
    let inputCopyMS = (ProcessInfo.processInfo.systemUptime - copyStart) * 1000
    let completed = try context.render(pixels, options: options, timestamp: timestamp)
    let outputCopyStart = ProcessInfo.processInfo.systemUptime
    let result = completed.pixels.buffer
    guard CVPixelBufferLockBaseAddress(result, .readOnly) == kCVReturnSuccess else {
      throw MLXMediaError("Cannot map NR output")
    }
    defer { CVPixelBufferUnlockBaseAddress(result, .readOnly) }
    guard let bytes = CVPixelBufferGetBaseAddress(result) else {
      throw MLXMediaError("Empty NR output")
    }
    for y in 0..<Int(height) {
      memcpy(
        output.advanced(by: y * Int(outputPitch)),
        bytes.advanced(by: y * CVPixelBufferGetBytesPerRow(result)), Int(width) * 4)
    }
    context.profile = completed.profile + [inputCopyMS, (ProcessInfo.processInfo.systemUptime - outputCopyStart) * 1000]
    context.error = ""
    return true
  } catch {
    context.error = String(describing: error)
    return false
  }
}

// Native source entry point: retain IOSurfaces across the synchronous worker
// call instead of copying full frames into and out of the row-pitch C ABI.
// The returned +1 CVPixelBuffer belongs to the caller. The row-pitch API remains
// available for the upstream OBS effect filter, whose staging buffer is CPU-owned.
@_cdecl("nr5_process_surface")
public func nr5ProcessSurface(
  _ handle: UnsafeMutableRawPointer?, _ input: UnsafeMutableRawPointer?,
  _ settings: UnsafePointer<CChar>?, _ timestamp: UInt64
) -> UnsafeMutableRawPointer? {
  guard let handle, let input, let settings else { return nil }
  let context = Unmanaged<NativeContext>.fromOpaque(handle).takeUnretainedValue()
  context.lock.lock()
  defer { context.lock.unlock() }
  do {
    let pixels = Unmanaged<CVPixelBuffer>.fromOpaque(input).takeUnretainedValue()
    guard CVPixelBufferGetPixelFormatType(pixels) == kCVPixelFormatType_32BGRA,
      CVPixelBufferGetIOSurface(pixels) != nil,
      (1...8192).contains(CVPixelBufferGetWidth(pixels)),
      (1...8192).contains(CVPixelBufferGetHeight(pixels))
    else { throw MLXMediaError("Native NR requires an owned BGRA IOSurface") }
    let options = try JSONDecoder().decode(Options.self, from: Data(String(cString: settings).utf8))
    let completed = try context.render(pixels, options: options, timestamp: timestamp)
    context.profile = completed.profile + [0, 0]
    context.error = ""
    return Unmanaged.passRetained(completed.pixels.buffer).toOpaque()
  } catch {
    context.error = String(describing: error)
    return nil
  }
}
