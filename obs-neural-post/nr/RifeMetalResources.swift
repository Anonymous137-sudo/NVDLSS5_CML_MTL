import Darwin
import Foundation

// SwiftPM's default lookup starts at the host app. Resolve the shader bundle
// relative to the dylib so a stock OBS installation can load the plugin.
extension Bundle {
    static var module: Bundle {
        var image = Dl_info()
        let anchor: @convention(c) () -> Void = rifeMetalImageAnchor
        precondition(dladdr(unsafeBitCast(anchor, to: UnsafeRawPointer.self), &image) != 0)
        let resources = URL(fileURLWithPath: String(cString: image.dli_fname))
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Resources")
        return Bundle(url: resources.appendingPathComponent("RifeMetal_RifeMetalCore.bundle"))!
    }
}

private func rifeMetalImageAnchor() {}
