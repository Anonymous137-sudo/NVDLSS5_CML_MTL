// Real HDMI capture audit and viewer. Compile standalone for raw Metal preview,
// or as the PS5NeuralAudit SwiftPM product for the shipped MLX reconstruction path.
import AppKit
import AVFoundation
import CoreImage
import CoreMedia
import ImageIO
import UniformTypeIdentifiers
import MetalKit
#if canImport(NeuralPost)
import NeuralPost

// Uses exactly the shipped reconstruction runtime; capture never waits for it.
final class LiveNeuralRun {
    let pipeline:PostPipeline
    let presenter:NativePresenter
    let destination:URL
    let lock = NSLock()
    let exports = DispatchQueue(label:"capture.audit.neural-exports")
    var inputs:[UInt64:CVPixelBuffer] = [:]
    var times:[Double] = [], outputAges:[Double] = [], aneTimes:[Double] = []
    var completed = 0, exportedPairs = 0
    var outputWidth = 0, outputHeight = 0
    init(destination:URL,resources:URL) throws {
        self.destination = destination
        resources.appendingPathComponent("mlx.metallib").path.withCString { npSetMetallib($0) }
        let aneURL=CommandLine.arguments.first(where:{$0.hasPrefix("--ane-model=")}).map { URL(fileURLWithPath:String($0.dropFirst(12))) }
        pipeline = try PostPipeline(modelURL:resources.appendingPathComponent("rt4ksr-x3.safetensors"),aneURL:aneURL)
        presenter = try NativePresenter(pipeline:pipeline)
        pipeline.onComplete = { [weak self] frame in self?.complete(frame) }
    }
    func submit(_ buffer:CVPixelBuffer,sequence:UInt64,pts:Double) {
        lock.lock()
        inputs[sequence] = buffer
        while inputs.count>4, let oldest = inputs.keys.min() { inputs.removeValue(forKey:oldest) }
        lock.unlock()
        pipeline.submit(buffer,sequence:sequence,sourceSeconds:pts)
    }
    func complete(_ frame:ProcessedFrame) {
        lock.lock()
        completed += 1
        outputWidth = CVPixelBufferGetWidth(frame.pixels); outputHeight = CVPixelBufferGetHeight(frame.pixels)
        if completed>30 {
            times.append(frame.processingMS)
            if frame.aneMS>0 { aneTimes.append(frame.aneMS); if aneTimes.count>18000 { aneTimes.removeFirst() } }
            outputAges.append((ProcessInfo.processInfo.systemUptime-frame.sourceSeconds)*1000)
            if times.count>18000 { times.removeFirst(); outputAges.removeFirst() }
        }
        let input = [60,600,1800].contains(completed) ? inputs[frame.sequence] : nil
        lock.unlock()
        if let input {
            exports.async { [self,frame,input] in
                let ci = CIContext(options:[.cacheIntermediates:false])
                var written = 0
                for (name,buffer) in [("input",input),("neural-4k",frame.pixels)] {
                    let image = CIImage(cvPixelBuffer:buffer)
                    guard let cg = ci.createCGImage(image,from:image.extent), let writer = CGImageDestinationCreateWithURL(destination.appendingPathComponent("pair-\(frame.sequence)-\(name).png") as CFURL,UTType.png.identifier as CFString,1,nil) else { continue }
                    CGImageDestinationAddImage(writer,cg,nil); if CGImageDestinationFinalize(writer) { written += 1 }
                }
                lock.lock(); if written == 2 { exportedPairs += 1 }; lock.unlock()
            }
        }
    }
    func resetCapture() {
        pipeline.reset()
        lock.lock(); inputs.removeAll(); lock.unlock()
    }
    func stop() {
        pipeline.stop()
        let deadline = ProcessInfo.processInfo.systemUptime+5
        while pipeline.stats.busy && ProcessInfo.processInfo.systemUptime<deadline { Thread.sleep(forTimeInterval:0.001) }
        exports.sync {}
        lock.lock(); inputs.removeAll(); lock.unlock()
    }
    func report(seconds:Double)->[String:Any] {
        let s = pipeline.stats, d = presenter.displayStats
        lock.lock(); let t = times.sorted(), a = outputAges.sorted(), ane = aneTimes.sorted(), pairs = exportedPairs, w = outputWidth, h = outputHeight; lock.unlock()
        func p(_ v:[Double],_ q:Double)->Double { v.isEmpty ? 0 : v[min(v.count-1,Int(Double(v.count-1)*q))] }
        return ["submitted":s.submitted,"completed":s.completed,"neuralCompleted":s.neuralCompleted,
                "superseded":s.superseded,"failed":s.failed,"error":s.error,
                "completionFPS":Double(s.completed)/max(0.001,seconds),"warmSamples":t.count,
                "processingP50MS":p(t,0.5),"processingP95MS":p(t,0.95),"processingP99MS":p(t,0.99),
                "sourceToCompletionP50MS":p(a,0.5),"sourceToCompletionP95MS":p(a,0.95),
                "outputWidth":w,"outputHeight":h,"matchedExportPairs":pairs,
                "presentedDrawables":d.count,"uniquePresentedSourceSequences":d.unique,
                "drawablePresentationRateHz":d.rate,"displayActiveSeconds":d.lastTime-d.firstTime,
                "displayGPUFailures":d.errors,"ane":!ane.isEmpty,"appearanceCompleted":s.appearanceCompleted,"anePreparationPredictionTransferP50MS":p(ane,0.5),"anePreparationPredictionTransferP95MS":p(ane,0.95),
                "note":"Real HDMI through shipped MLX/Metal runtime and native preview; not OBS composition. First 30 completions excluded from timing percentiles. PNG exports can perturb timing. Display repeats are not additional reconstructed frames."]
    }
}
#endif

// Retain only the newest capture buffer and at most two in-flight GPU reads.
// This avoids retaining capture-pool buffers in the sample-buffer display queue.
final class RawMetalView: MTKView, MTKViewDelegate {
    private let captureLock = NSLock()
    private var newest:CVPixelBuffer?
    private let commands:MTLCommandQueue
    private let renderState:MTLRenderPipelineState
    private let slots = DispatchSemaphore(value:2)
    init(captureSize:CGSize) throws {
        guard let gpu = MTLCreateSystemDefaultDevice(), let commands = gpu.makeCommandQueue() else {
            throw NSError(domain:"PS5Capture",code:1,userInfo:[NSLocalizedDescriptionKey:"Metal unavailable"])
        }
        self.commands = commands
        let library = try gpu.makeLibrary(source:"""
        #include <metal_stdlib>
        using namespace metal;
        struct V { float4 p [[position]]; float2 uv; };
        vertex V vertex_main(uint i [[vertex_id]]) {
            float2 p=float2((i<<1)&2,i&2); return {float4(p*2-1,0,1),float2(p.x,1-p.y)};
        }
        fragment float4 fragment_main(V v [[stage_in]],texture2d<float> t [[texture(0)]]) {
            constexpr sampler s(filter::linear,address::clamp_to_edge); return t.sample(s,v.uv);
        }
        """,options:nil)
        let description = MTLRenderPipelineDescriptor()
        description.vertexFunction = library.makeFunction(name:"vertex_main")
        description.fragmentFunction = library.makeFunction(name:"fragment_main")
        description.colorAttachments[0].pixelFormat = .bgra8Unorm
        renderState = try gpu.makeRenderPipelineState(descriptor:description)
        super.init(frame:NSRect(x:0,y:0,width:960,height:540),device:gpu)
        colorPixelFormat = .bgra8Unorm; framebufferOnly = true
        preferredFramesPerSecond = 120; delegate = self
        clearColor = MTLClearColorMake(0,0,0,1)
    }
    required init(coder:NSCoder) { fatalError("Use init()") }
    func submit(_ buffer:CVPixelBuffer) { captureLock.lock(); newest = buffer; captureLock.unlock() }
    func stop() { isPaused = true; captureLock.lock(); newest = nil; captureLock.unlock() }
    func mtkView(_ view:MTKView,drawableSizeWillChange size:CGSize) {}
    func draw(in view:MTKView) {
        captureLock.lock(); let buffer = newest; captureLock.unlock()
        guard let buffer, let gpu = device, slots.wait(timeout:.now()) == .success else { return }
        guard let surface = CVPixelBufferGetIOSurface(buffer)?.takeUnretainedValue(),
              let drawable = currentDrawable, let pass = currentRenderPassDescriptor,
              let cb = commands.makeCommandBuffer() else { slots.signal(); return }
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.bgra8Unorm,width:CVPixelBufferGetWidth(buffer),height:CVPixelBufferGetHeight(buffer),mipmapped:false)
        td.storageMode = .shared; td.usage = .shaderRead
        guard let texture = gpu.makeTexture(descriptor:td,iosurface:surface,plane:0),
              let encoder = cb.makeRenderCommandEncoder(descriptor:pass) else { slots.signal(); return }
        let width = Double(drawable.texture.width), height = Double(drawable.texture.height)
        let scale = min(width/Double(texture.width),height/Double(texture.height))
        let w = Double(texture.width)*scale, h = Double(texture.height)*scale
        encoder.setViewport(MTLViewport(originX:(width-w)/2,originY:(height-h)/2,width:w,height:h,znear:0,zfar:1))
        encoder.setRenderPipelineState(renderState); encoder.setFragmentTexture(texture,index:0)
        encoder.drawPrimitives(type:.triangle,vertexStart:0,vertexCount:3); encoder.endEncoding()
        cb.present(drawable)
        let slots = self.slots
        cb.addCompletedHandler { [buffer,texture] _ in _ = buffer; _ = texture; slots.signal() }
        cb.commit()
    }
}

final class CaptureAudit: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, NSApplicationDelegate {
    let session = AVCaptureSession()
    let control = DispatchQueue(label:"capture.audit.control")
    let capture = DispatchQueue(label:"capture.audit.frames")
    let snapshots = DispatchQueue(label:"capture.audit.snapshots")
    let lock = NSLock()
    let ci = CIContext(options:[.cacheIntermediates:false])
    let destination:URL
    #if canImport(NeuralPost)
    var neural:LiveNeuralRun?
    #endif
    var window:NSWindow?
    var timer:Timer?
    var frames = 0, dropped = 0
    var firstArrival:Double = 0, lastArrival:Double = 0, lastPTS:Double = 0
    var intervals:[Double] = [], ages:[Double] = [], lumas:[Double] = []
    var deviceName = "", actualFormat = "", snapshotCount = 0
    var selectedFormat = -1, selectedFPS:Double = 0
    var deliveredWidth = 0, deliveredHeight = 0
    var negotiated:[String:Any] = [:]
    let simulateStall = CommandLine.arguments.contains("--simulate-stall-once")
    var stallRecoveries = 0
    var recoveryBusy = false
    var lastRecovery:Double = 0
    var finished = false
    var endedAt:Double = 0
    var beganAt:Double = 0
    var rawView:RawMetalView?
    let duration:Double = CommandLine.arguments.contains("--live") ? .infinity : (CommandLine.arguments.first(where:{$0.hasPrefix("--seconds=")}).flatMap { Double($0.dropFirst(10)) } ?? 15)
    func applicationShouldTerminateAfterLastWindowClosed(_ sender:NSApplication)->Bool { true }
    func applicationWillTerminate(_ notification:Notification) {
        timer?.invalidate()
        control.sync {
            session.stopRunning(); capture.sync {}; snapshots.sync {}
            #if canImport(NeuralPost)
            neural?.stop()
            #endif
        }
        #if canImport(NeuralPost)
        neural?.presenter.stop()
        #endif
        rawView?.stop()
        if !finished { endedAt = ProcessInfo.processInfo.systemUptime; write(["status":"closed"]) }
    }
    init(destination:URL) { self.destination = destination }
    func write(_ extra:[String:Any]) {
        lock.lock()
        let f = frames, d = dropped, first = firstArrival, last = lastArrival
        let dt = intervals.sorted(), age = ages.sorted(), luma = lumas
        let snapshots = snapshotCount
        let width = deliveredWidth, height = deliveredHeight, recoveries = stallRecoveries
        lock.unlock()
        func p(_ a:[Double],_ q:Double)->Double { a.isEmpty ? 0 : a[min(a.count-1,Int(Double(a.count-1)*q))] }
        var data:[String:Any] = ["device":deviceName,"frames":f,"captureReportedDrops":d,"stallRecoveries":recoveries,"simulatedStall":simulateStall,
            "observedFPS":first>0 ? Double(f)/max(0.001,(endedAt>0 ? endedAt : ProcessInfo.processInfo.systemUptime)-first) : 0,
            "arrivalSpanFPS":last>first ? Double(max(0,f-1))/(last-first) : 0,
            "secondsSinceLastFrame":last>0 ? ProcessInfo.processInfo.systemUptime-last : 0,
            "captureStalled":last>0 && ProcessInfo.processInfo.systemUptime-last>2,
            "intervalP50MS":p(dt,0.5),"intervalP95MS":p(dt,0.95),
            "sourceAgeP50MS":p(age,0.5),"sourceAgeP95MS":p(age,0.95),
            "meanSampledLuma":luma.isEmpty ? 0 : luma.reduce(0,+)/Double(luma.count),
            "deliveredPixelFormat":actualFormat,"selectedFormatIndex":selectedFormat,
            "selectedFPS":selectedFPS,"snapshotCount":snapshots,
            "deliveredWidth":width,"deliveredHeight":height,"negotiated":negotiated,
            "previewBackend":"bounded Metal renderer",
            "percentileWindow":"Most recent 3600 intervals and source-age samples",
            "note":"Capture/source timestamp age is not input-to-photon latency. Neural results, when enabled, are reported separately. Still export may perturb isolated frame intervals."]
        #if canImport(NeuralPost)
        if let neural {
            let elapsed = first>0 ? (endedAt>0 ? endedAt : ProcessInfo.processInfo.systemUptime)-first : 0
            data["neural"] = neural.report(seconds:elapsed)
        }
        #endif
        extra.forEach { data[$0.key] = $0.value }
        if let bytes = try? JSONSerialization.data(withJSONObject:data,options:[.prettyPrinted,.sortedKeys]) {
            try? bytes.write(to:destination.appendingPathComponent("capture.json"),options:.atomic)
        }
    }
    func start() {
        beganAt = ProcessInfo.processInfo.systemUptime
        try? FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true)
        write(["status":"awaiting-camera-permission"])
        let view:NSView
        do { let raw = try RawMetalView(captureSize:CGSize(width:960,height:540)); rawView = raw; view = raw }
        catch { fail(String(describing:error)); return }
        let window = NSWindow(contentRect:view.bounds,styleMask:[.titled,.closable,.resizable,.miniaturizable],backing:.buffered,defer:false)
        window.title = "PS5 · live HDMI · starting capture"
        window.contentView = view; window.isReleasedWhenClosed = false
        #if canImport(NeuralPost)
        if let resourceArg = CommandLine.arguments.first(where:{$0.hasPrefix("--neural-resources=")}) {
            do {
                let resources = URL(fileURLWithPath:String(resourceArg.dropFirst(19)))
                let run = try LiveNeuralRun(destination:destination,resources:resources)
                neural = run; rawView?.stop(); rawView = nil; window.contentView = run.presenter
            } catch { fail("Neural runtime initialization failed: \(error)"); return }
        }
        #endif
        window.collectionBehavior = .fullScreenPrimary
        window.center(); window.makeKeyAndOrderFront(nil); self.window = window
        switch AVCaptureDevice.authorizationStatus(for:.video) {
        case .authorized: configure()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for:.video) { allowed in
                if allowed { self.configure() } else { self.fail("Camera permission denied") }
            }
        default: fail("Camera permission unavailable; enable it in macOS Privacy & Security")
        }
    }
    func fail(_ message:String) {
        finished = true; endedAt = ProcessInfo.processInfo.systemUptime
        write(["status":"failed","error":message])
        DispatchQueue.main.async { NSApplication.shared.terminate(nil) }
    }
    func configure() {
        control.async {
            do {
                let discovery = AVCaptureDevice.DiscoverySession(deviceTypes:[.external],mediaType:.video,position:.unspecified)
                guard let device = discovery.devices.first(where:{$0.localizedName == "USB3 Video"}) else {
                    self.fail("USB3 Video capture device not found"); return
                }
                self.deviceName = device.localizedName
                let input = try AVCaptureDeviceInput(device:device)
                self.session.beginConfiguration()
                guard self.session.canAddInput(input) else { self.session.commitConfiguration(); self.fail("Cannot add input"); return }
                self.session.addInput(input)
                if self.session.canSetSessionPreset(.hd1280x720) { self.session.sessionPreset = .hd1280x720 }
                var chosen:(Int,AVCaptureDevice.Format,AVFrameRateRange)?
                for (i,format) in device.formats.enumerated() {
                    let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                    if dimensions.width != 1280 || dimensions.height != 720 { continue }
                    if let range = format.videoSupportedFrameRateRanges.first(where:{$0.maxFrameRate>=59.9 && $0.minFrameRate<=60.01}) {
                        chosen = (i,format,range)
                        // Prefer native 420v if available; request BGRA from AVFoundation below.
                        if CMFormatDescriptionGetMediaSubType(format.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange { break }
                    }
                }
                guard let chosen else { self.session.commitConfiguration(); self.fail("No 720p60 mode"); return }
                self.selectedFormat = chosen.0; self.selectedFPS = chosen.2.maxFrameRate
                let output = AVCaptureVideoDataOutput()
                output.alwaysDiscardsLateVideoFrames = true
                output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA,
                                       kCVPixelBufferMetalCompatibilityKey as String:true,
                                       kCVPixelBufferWidthKey as String:1280,kCVPixelBufferHeightKey as String:720]
                output.setSampleBufferDelegate(self,queue:self.capture)
                guard self.session.canAddOutput(output) else { self.session.commitConfiguration(); self.fail("Cannot add output"); return }
                self.session.addOutput(output)
                if let connection = output.connection(with:.video) {
                    if connection.isVideoMinFrameDurationSupported { connection.videoMinFrameDuration = chosen.2.minFrameDuration }
                    if connection.isVideoMaxFrameDurationSupported { connection.videoMaxFrameDuration = chosen.2.minFrameDuration }
                }
                self.session.commitConfiguration()
                try device.lockForConfiguration()
                device.activeFormat = chosen.1
                device.activeVideoMinFrameDuration = chosen.2.minFrameDuration
                device.activeVideoMaxFrameDuration = chosen.2.minFrameDuration
                device.unlockForConfiguration()
                self.session.startRunning()
                let active = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
                self.negotiated = ["width":active.width,"height":active.height, "minFrameDurationSeconds":CMTimeGetSeconds(device.activeVideoMinFrameDuration), "maxFrameDurationSeconds":CMTimeGetSeconds(device.activeVideoMaxFrameDuration)]
                self.write(["status":"capturing"])
                DispatchQueue.main.async {
                    self.timer = Timer.scheduledTimer(withTimeInterval:1,repeats:true) { _ in self.tick() }
                }
            } catch { self.fail(String(describing:error)) }
        }
    }
    func tick() {
        lock.lock(); let first = firstArrival, last = lastArrival; lock.unlock()
        let now = ProcessInfo.processInfo.systemUptime
        if first==0 && now-beganAt>10 { fail("No capture frames arrived within ten seconds"); return }
        if first>0 && now-last>2 && !recoveryBusy && now-lastRecovery>5 {
            guard stallRecoveries<3 else { fail("Capture stalled after three bounded recovery attempts"); return }
            recoveryBusy = true; lock.lock(); stallRecoveries += 1; lock.unlock(); lastRecovery = now
            control.async {
                self.session.stopRunning(); self.capture.sync {}
                #if canImport(NeuralPost)
                self.neural?.resetCapture()
                #endif
                self.session.startRunning()
                DispatchQueue.main.async { self.recoveryBusy = false }
            }
        }
        if first>0 && ProcessInfo.processInfo.systemUptime-first>=duration {
            timer?.invalidate()
            control.async {
                self.session.stopRunning()
                self.capture.sync {}
                self.snapshots.sync {}
                #if canImport(NeuralPost)
                self.neural?.stop()
                #endif
                self.endedAt = ProcessInfo.processInfo.systemUptime
                self.finished = true
                self.write(["status":"complete","requestedSeconds":self.duration])
                DispatchQueue.main.async { self.window?.close(); NSApplication.shared.terminate(nil) }
            }
        } else {
            write(["status":"capturing"])
            lock.lock(); let count = frames, elapsed = lastArrival-firstArrival, w = deliveredWidth, h = deliveredHeight; lock.unlock()
            #if canImport(NeuralPost)
            if neural != nil { return }
            #endif
            window?.title = String(format:"PS5 · %d×%d · %.1f capture FPS · raw HDMI",w,h,elapsed>0 ? Double(count)/max(0.001,ProcessInfo.processInfo.systemUptime-first) : 0)
        }
    }
    func captureOutput(_ output:AVCaptureOutput,didDrop sample:CMSampleBuffer,from connection:AVCaptureConnection) {
        lock.lock(); dropped += 1; lock.unlock()
    }
    func captureOutput(_ output:AVCaptureOutput,didOutput sample:CMSampleBuffer,from connection:AVCaptureConnection) {
        // Opt-in watchdog validation only; real launchers never enable injection.
        lock.lock(); let suppress = simulateStall && frames>=120 && stallRecoveries==0; lock.unlock()
        if suppress { return }
        guard let buffer = CMSampleBufferGetImageBuffer(sample) else { return }
        var showRaw = true
        #if canImport(NeuralPost)
        showRaw = neural == nil
        #endif
        if showRaw { rawView?.submit(buffer) }
        let now = ProcessInfo.processInfo.systemUptime
        var pts = CMSampleBufferGetPresentationTimeStamp(sample)
        if let clock = session.synchronizationClock, clock != CMClockGetHostTimeClock() {
            pts = CMSyncConvertTime(pts,from:clock,to:CMClockGetHostTimeClock())
        }
        let age = (now-CMTimeGetSeconds(pts))*1000
        #if canImport(NeuralPost)
        if let neural, CMTimeGetSeconds(pts).isFinite {
            lock.lock(); let next = frames+1; lock.unlock()
            neural.submit(buffer,sequence:UInt64(next),pts:CMTimeGetSeconds(pts))
        }
        #endif
        lock.lock()
        frames += 1; let sequence = frames
        if firstArrival==0 { firstArrival = now }
        if lastArrival>0 { intervals.append((now-lastArrival)*1000) }
        lastArrival = now
        if age.isFinite { ages.append(age) }
        if intervals.count > 3600 { intervals.removeFirst(intervals.count-3600) }
        if ages.count > 3600 { ages.removeFirst(ages.count-3600) }
        let format = CVPixelBufferGetPixelFormatType(buffer)
        deliveredWidth = CVPixelBufferGetWidth(buffer); deliveredHeight = CVPixelBufferGetHeight(buffer)
        actualFormat = String(format:"0x%08x",format)
        lock.unlock()
        if format == kCVPixelFormatType_32BGRA && sequence%30==0 {
            CVPixelBufferLockBaseAddress(buffer,.readOnly)
            if let base = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to:UInt8.self) {
                let row = CVPixelBufferGetBytesPerRow(buffer)
                var total = 0.0
                for y in 0..<9 { for x in 0..<16 {
                    let p = ((y*2+1)*CVPixelBufferGetHeight(buffer)/18)*row+((x*2+1)*CVPixelBufferGetWidth(buffer)/32)*4
                    total += (Double(base[p])*0.0722+Double(base[p+1])*0.7152+Double(base[p+2])*0.2126)/255
                } }
                lock.lock(); lumas.append(total/144); if lumas.count > 120 { lumas.removeFirst(lumas.count-120) }; lock.unlock()
            }
            CVPixelBufferUnlockBaseAddress(buffer,.readOnly)
        }
        if [30,300,600].contains(sequence) {
            snapshots.async { [buffer] in
                let image = CIImage(cvPixelBuffer:buffer)
                guard let cg = self.ci.createCGImage(image,from:image.extent),
                      let writer = CGImageDestinationCreateWithURL(self.destination.appendingPathComponent("frame-\(sequence).png") as CFURL,UTType.png.identifier as CFString,1,nil) else { return }
                CGImageDestinationAddImage(writer,cg,nil)
                if CGImageDestinationFinalize(writer) { self.lock.lock(); self.snapshotCount += 1; self.lock.unlock() }
            }
        }
    }
}

@main struct Main {
    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.regular)
        let path = CommandLine.arguments.dropFirst().first ?? "/tmp/obs-neural-capture-audit"
        let audit = CaptureAudit(destination:URL(fileURLWithPath:path))
        app.delegate = audit
        audit.start(); app.activate(ignoringOtherApps:true)
        withExtendedLifetime(audit) { app.run() }
    }
}
