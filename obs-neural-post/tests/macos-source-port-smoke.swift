import CoreVideo
import Darwin
import Foundation

let plugin = URL(fileURLWithPath: CommandLine.arguments[1])
let library = plugin.appendingPathComponent("Contents/Frameworks/libnvidia_nr.dylib")
guard let module = dlopen(library.path, RTLD_NOW | RTLD_LOCAL) else {
  fatalError(String(cString: dlerror()))
}
func symbol<T>(_ name: String, as type: T.Type) -> T {
  guard let address = dlsym(module, name) else { fatalError("Missing symbol \(name)") }
  return unsafeBitCast(address, to: T.self)
}
typealias Create = @convention(c) (UnsafePointer<CChar>) -> UnsafeMutableRawPointer?
typealias Submit =
  @convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?, UInt64, Double) -> Void
typealias Acquire = @convention(c) (UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer?
typealias Pixels = @convention(c) (UnsafeMutableRawPointer) -> UnsafeMutableRawPointer
typealias Release = @convention(c) (UnsafeMutableRawPointer?) -> Void
typealias Status =
  @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<CChar>, Int32) -> Void
let create = symbol("nr_create", as: Create.self)
let submit = symbol("nr_submit", as: Submit.self)
let acquire = symbol("nr_frame_acquire", as: Acquire.self)
let pixels = symbol("nr_frame_pixels", as: Pixels.self)
let release = symbol("nr_frame_release", as: Release.self)
let destroy = symbol("nr_destroy", as: Release.self)
let status = symbol("nr_status", as: Status.self)
let resources = plugin.appendingPathComponent("Contents/Resources")
let settings: [String: Any] = [
  "mode": "31-480", "enabled": true, "upscaling": true,
  "style": 0, "intensityMode": "final", "intensity": 1,
  "tone": 1, "structure": 1, "skinStructure": -1, "maskStructure": -1,
  "automaticMask": false, "noiseFrame": 0,
  "model71": resources.appendingPathComponent("NeuralRendering.dlssmodel").path,
  "model31": resources.appendingPathComponent("NeuralRendering-Small.dlssmodel").path,
  "srPath": resources.appendingPathComponent("rt4ksr-x3.safetensors").path,
  "frameGenerationBackend": "rife", "frameGenerationFactor": 1, "frameGenerationHeight": 360,
  "frameGenerationPath": resources.appendingPathComponent("framegen.safetensors").path,
  "captureGuided": true, "temporalNR": false, "nativeProcessingHeight": 480,
]
let text = String(decoding: try JSONSerialization.data(withJSONObject: settings), as: UTF8.self)
guard let session = text.withCString({ create($0) }) else {
  fatalError("Source bridge creation failed")
}
defer { destroy(session) }
var buffer: CVPixelBuffer?
let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary
precondition(
  CVPixelBufferCreate(nil, 320, 320, kCVPixelFormatType_32BGRA, attrs, &buffer) == kCVReturnSuccess)
let input = buffer!
CVPixelBufferLockBaseAddress(input, [])
let bytes = CVPixelBufferGetBaseAddress(input)!.assumingMemoryBound(to: UInt8.self)
let pitch = CVPixelBufferGetBytesPerRow(input)
for y in 0..<320 {
  for x in 0..<320 {
    bytes[y * pitch + x * 4] = UInt8((x * 3 + y) % 256)
    bytes[y * pitch + x * 4 + 1] = UInt8((y * 2) % 256)
    bytes[y * pitch + x * 4 + 2] = UInt8(x % 256)
    bytes[y * pitch + x * 4 + 3] = 255
  }
}
CVPixelBufferUnlockBaseAddress(input, [])
submit(session, Unmanaged.passUnretained(input).toOpaque(), 1, ProcessInfo.processInfo.systemUptime)
let deadline = Date().addingTimeInterval(30)
var result: UnsafeMutableRawPointer?
repeat {
  result = acquire(session)
  if result == nil { Thread.sleep(forTimeInterval: 0.05) }
} while result == nil && Date() < deadline
var message = [CChar](repeating: 0, count: 200_000)
status(session, &message, Int32(message.count))
let report =
  try JSONSerialization.jsonObject(with: Data(String(cString: message).utf8)) as! [String: Any]
print("SOURCE_PORT_STATUS", String(cString: message))
precondition((report["failed"] as? Int) == 0)
precondition((report["captureGuided"] as? Bool) == false)
precondition(report["nrImplementation"] as? String == "obs-dlss5-nr-macos")
let stages = report["stages"] as! [String: Any]
precondition((stages["upstreamNativeNR"] as? Int) == 1)
precondition((stages["nativeSurfaceBridge"] as? Int) == 1)
precondition((stages["nativeInputCopyMS"] as? Double) == 0)
precondition((stages["nativeOutputCopyMS"] as? Double) == 0)
precondition((stages["nativePass1MS"] as? Double ?? 0) > 0)
guard let result else { fatalError("No model output before deadline") }
defer { release(result) }
let output = Unmanaged<CVPixelBuffer>.fromOpaque(pixels(result)).takeUnretainedValue()
precondition(CVPixelBufferGetWidth(output) == 320 && CVPixelBufferGetHeight(output) == 320)
print("SOURCE_PORT_PASS")
