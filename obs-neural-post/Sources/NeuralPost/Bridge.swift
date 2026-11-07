import AppKit
import CoreVideo
import Foundation
import MLX

@_cdecl("np_set_metallib")
public func npSetMetallib(_ path: UnsafePointer<CChar>) {
    GPU.metallib = URL(fileURLWithPath:String(cString:path))
}

private final class Bridge {
    let pipeline: PostPipeline
    var window: NSWindow?
    init(path: String, appearance: String, anePath:String = "") throws {
        pipeline = try PostPipeline(modelURL:path.isEmpty ? nil : URL(fileURLWithPath:path),
                                    appearanceURL:appearance.isEmpty ? nil : URL(fileURLWithPath:appearance),aneURL:anePath.isEmpty ? nil : URL(fileURLWithPath:anePath))
    }
}

@_cdecl("np_create")
public func npCreate(_ path: UnsafePointer<CChar>?, _ appearance: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
    do { return Unmanaged.passRetained(try Bridge(path:path.map(String.init(cString:)) ?? "", appearance:appearance.map(String.init(cString:)) ?? "")).toOpaque() }
    catch { fputs("NeuralPost init: \(error)\n",stderr); return nil }
}
@_cdecl("np_create_hybrid")
public func npCreateHybrid(_ path:UnsafePointer<CChar>?,_ ane:UnsafePointer<CChar>?)->UnsafeMutableRawPointer? {
    do { return Unmanaged.passRetained(try Bridge(path:path.map(String.init(cString:)) ?? "",appearance:"",anePath:ane.map(String.init(cString:)) ?? "")).toOpaque() }
    catch { fputs("NeuralPost hybrid: \(error)\n",stderr); return nil }
}
@_cdecl("np_destroy")
public func npDestroy(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    let bridge = Unmanaged<Bridge>.fromOpaque(handle).takeRetainedValue()
    bridge.pipeline.stop()
    DispatchQueue.main.async { bridge.window?.close(); bridge.window = nil }
}
@_cdecl("np_submit")
public func npSubmit(_ handle: UnsafeMutableRawPointer?, _ buffer: UnsafeMutableRawPointer?, _ sequence: UInt64, _ seconds: Double) {
    guard let handle, let buffer else { return }
    let bridge = Unmanaged<Bridge>.fromOpaque(handle).takeUnretainedValue()
    let pixels = Unmanaged<CVPixelBuffer>.fromOpaque(buffer).takeUnretainedValue()
    bridge.pipeline.submit(pixels,sequence: sequence,sourceSeconds: seconds)
}
@_cdecl("np_reset")
public func npReset(_ handle: UnsafeMutableRawPointer?) {
    if let handle { Unmanaged<Bridge>.fromOpaque(handle).takeUnretainedValue().pipeline.reset() }
}
// Caller owns this frame lease until np_frame_release. Pixel buffer remains
// immutable and alive independently of replacement by the inference worker.
@_cdecl("np_frame_acquire")
public func npFrameAcquire(_ handle: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let handle, let frame = Unmanaged<Bridge>.fromOpaque(handle).takeUnretainedValue().pipeline.latest() else { return nil }
    return Unmanaged.passRetained(frame).toOpaque()
}
@_cdecl("np_frame_pixels")
public func npFramePixels(_ frame: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer {
    Unmanaged.passUnretained(Unmanaged<ProcessedFrame>.fromOpaque(frame).takeUnretainedValue().pixels).toOpaque()
}
@_cdecl("np_frame_release")
public func npFrameRelease(_ frame: UnsafeMutableRawPointer?) {
    if let frame { Unmanaged<ProcessedFrame>.fromOpaque(frame).release() }
}
@_cdecl("np_show_preview")
public func npShowPreview(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    let bridge = Unmanaged<Bridge>.fromOpaque(handle).takeUnretainedValue()
    DispatchQueue.main.async {
        if bridge.window == nil {
            do {
                let view = try NativePresenter(pipeline: bridge.pipeline)
                let window = NSWindow(contentRect: view.bounds,styleMask: [.titled,.closable,.resizable,.miniaturizable],backing: .buffered,defer: false)
                window.title = "Neural Post · native display cadence · MLX + Metal"
                window.isReleasedWhenClosed = false; window.contentView = view; window.center()
                bridge.window = window
            } catch { fputs("Preview: \(error)\n",stderr) }
        }
        bridge.window?.makeKeyAndOrderFront(nil)
    }
}
@_cdecl("np_status")
public func npStatus(_ handle: UnsafeMutableRawPointer?, _ text: UnsafeMutablePointer<CChar>?, _ count: Int32) {
    guard let handle, let text, count > 0 else { return }
    let p = Unmanaged<Bridge>.fromOpaque(handle).takeUnretainedValue().pipeline
    let s = p.stats, frame = p.latest()
    let message = "MLX GPU; appearance=\(s.appearanceCompleted) ane_ms=\(frame?.aneMS ?? 0) completed=\(s.completed) neural=\(s.neuralCompleted) replaced=\(s.superseded) failed=\(s.failed) processing_ms=\(frame?.processingMS ?? 0) error=\(s.error)"
    message.withCString { _ = strlcpy(text,$0,Int(count)) }
}
