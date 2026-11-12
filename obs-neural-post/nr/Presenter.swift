import AppKit
import Metal
import QuartzCore

import DLSSRTSS

/// Independent of OBS output FPS. Repeats the latest finished image on the
/// display's cadence; this is presentation, never frame interpolation.
public final class NativePresenter: NSView, CAMetalDisplayLinkDelegate {
    private let pipeline: PostPipeline
    private let metalLayer = CAMetalLayer()
    private let commands: MTLCommandQueue
    private let renderState: MTLRenderPipelineState
    private var link: CAMetalDisplayLink?
    private let slots = DispatchSemaphore(value: 2)
    public private(set) var callbacks: UInt64 = 0
    public private(set) var presented: UInt64 = 0
    public private(set) var uniquePresented: UInt64 = 0
    public private(set) var lastSequence: UInt64 = 0
    public private(set) var lastSourceAgeMS: Double = 0
    public private(set) var lastCallbackIntervalMS: Double = 0
    private var lastCallbackTime: Double = 0
    private var lastGeneration:UInt64=UInt64.max
    private var lastChosenFrame:ProcessedFrame?
    private var backtracksPrevented:UInt64=0
    private let measurements = DisplayMeasurements()
    private let statusHUD=NSTextField(labelWithString:"DLSS ON")
    private let statsHUD=NativeStatsOverlay(frame:NSRect(x:12,y:12,width:305,height:245))
    private var statsVisible=(UserDefaults.standard.object(forKey:"DLSS.NativeStatsVisible") as? Bool) ?? true
    public var displayStats:DisplayPresentationStats { measurements.snapshot() }
    public var drawablePixels:CGSize { metalLayer.drawableSize }
    private var lastTelemetryTime:Double = 0
    private var previousNeural:UInt64 = 0
    private var previousAppearance:UInt64 = 0

    public init(pipeline: PostPipeline) throws {
        self.pipeline = pipeline
        guard let commands = pipeline.device.makeCommandQueue() else { throw NSError(domain: "NeuralPost", code: 3) }
        self.commands = commands
        let lib = try pipeline.device.makeLibrary(source: """
        #include <metal_stdlib>
        using namespace metal;
        struct V { float4 p [[position]]; float2 uv; };
        vertex V vertex_main(uint i [[vertex_id]]) {
          float2 p=float2((i<<1)&2,i&2); return {float4(p*2-1,0,1),float2(p.x,1-p.y)};
        }
        fragment float4 fragment_main(V v [[stage_in]],texture2d<float> t [[texture(0)]]) {
          constexpr sampler s(filter::linear,address::clamp_to_edge); return t.sample(s,v.uv);
        }
        """, options: nil)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = lib.makeFunction(name: "vertex_main")
        desc.fragmentFunction = lib.makeFunction(name: "fragment_main")
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        renderState = try pipeline.device.makeRenderPipelineState(descriptor: desc)
        super.init(frame: NSRect(x: 0,y: 0,width: 1280,height: 720))
        wantsLayer = true; layer = metalLayer
        metalLayer.device = pipeline.device; metalLayer.pixelFormat = .bgra8Unorm
        metalLayer.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
        metalLayer.framebufferOnly = true; metalLayer.maximumDrawableCount = 3
        statusHUD.font = .systemFont(ofSize:22,weight:.heavy);statusHUD.textColor = .white;statusHUD.backgroundColor = NSColor.black.withAlphaComponent(0.65);statusHUD.drawsBackground=true;statusHUD.isHidden=true;addSubview(statusHUD);addSubview(statsHUD);statsHUD.isHidden = !statsVisible
    }
    required init?(coder: NSCoder) { fatalError("Use init(pipeline:)") }
    public override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        link?.invalidate(); link = nil
        guard window != nil else { return }
        previousNeural=pipeline.stats.neuralCompleted;previousAppearance=pipeline.stats.appearanceCompleted
        window?.makeFirstResponder(self)
        let link = CAMetalDisplayLink(metalLayer: metalLayer)
        link.delegate = self
        link.preferredFrameLatency = 1
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 120, preferred: 120)
        link.add(to: .main, forMode: .common); self.link = link
    }
    public override func layout() {
        super.layout()
        let scale = window?.backingScaleFactor ?? 2
        statusHUD.frame=NSRect(x:statsVisible ? max(24,bounds.width-210):24,y:max(0,bounds.height-64),width:180,height:34)
        statsHUD.frame=NSRect(x:12,y:max(0,bounds.height-253),width:305,height:245)
        metalLayer.drawableSize = CGSize(width: max(1,bounds.width*scale), height: max(1,bounds.height*scale))
    }
    public func metalDisplayLink(_ link: CAMetalDisplayLink, needsUpdate update: CAMetalDisplayLink.Update) {
        callbacks += 1
        let overlay=pipeline.statusOverlayText;statusHUD.isHidden = overlay==nil;if let overlay,statusHUD.stringValue != overlay {statusHUD.stringValue=overlay}
        let generation=pipeline.presentationGeneration
        if generation != lastGeneration {measurements.resetWindow();lastChosenFrame=nil;lastGeneration=generation}
        let now = ProcessInfo.processInfo.systemUptime
        if lastCallbackTime > 0 { lastCallbackIntervalMS = (now-lastCallbackTime)*1000 }
        lastCallbackTime = now
        if now-lastTelemetryTime>=1 {
            let s = pipeline.stats, dt = lastTelemetryTime>0 ? now-lastTelemetryTime : 1
            let shown=displayStats
            if statsVisible {statsHUD.update(runtime:pipeline.status(),display:shown,sourceAge:lastSourceAgeMS)}
            pipeline.reportPresentation(["timelineBacktracksPrevented":backtracksPrevented,"active":shown.rate>0 && now-shown.lastTime<2,"lastDrawablePresentedAt":shown.lastTime,"sampledAtHostSeconds":now,"baseShownFPS":shown.baseShownFPS,"generatedShownFPS":shown.generatedShownFPS,"uniqueShownFPS":shown.freshShownFPS,"displayHz":shown.rate,"baseShown":shown.baseShown,"generatedShown":shown.generatedShown,"repeatedDrawables":shown.repeated,"gpuFailures":shown.errors,"uniqueIntervalP50MS":shown.uniqueIntervalP50MS,"uniqueIntervalP95MS":shown.uniqueIntervalP95MS,"uniqueIntervalP99MS":shown.uniqueIntervalP99MS,"uniqueIntervalMaxMS":shown.uniqueIntervalMaxMS,"holdsOver33MSLast256":shown.holdsOver33MS])
            window?.title = (pipeline.experimentalAppearance ? "Hybrid pilot · " : "") + pipeline.modeLabel + String(format:" · %.1f base + %.1f generated shown FPS · %.1f base processed FPS · %.1f display Hz · %.1f ms age",
                shown.baseShownFPS,shown.generatedShownFPS,Double(s.neuralCompleted-previousNeural)/dt,
                displayStats.rate,lastSourceAgeMS)
            previousNeural = s.neuralCompleted; previousAppearance = s.appearanceCompleted; lastTelemetryTime = now
        }
        guard var frame = pipeline.presentationFrame(at:update.targetPresentationTimestamp), slots.wait(timeout: .now()) == .success else { return }
        if let previous=lastChosenFrame,frame.presentationID<previous.presentationID {frame=previous;backtracksPrevented &+= 1}
        lastChosenFrame=frame
        guard let cb = commands.makeCommandBuffer() else { slots.signal(); return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = update.drawable.texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,1)
        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { slots.signal(); return }
        let width = Double(update.drawable.texture.width), height = Double(update.drawable.texture.height)
        let s = min(width/Double(frame.texture.width),height/Double(frame.texture.height)), w = Double(frame.texture.width)*s, h = Double(frame.texture.height)*s
        enc.setViewport(MTLViewport(originX: (width-w)/2,originY: (height-h)/2,width: w,height: h,znear: 0,zfar: 1))
        enc.setRenderPipelineState(renderState); enc.setFragmentTexture(frame.texture,index: 0)
        enc.drawPrimitives(type: .triangle,vertexStart: 0,vertexCount: 3); enc.endEncoding()
        cb.present(update.drawable)
        let slots = self.slots
        let measurements = self.measurements
        update.drawable.addPresentedHandler { [sequence=frame.presentationID] drawable in
            measurements.presented(sequence:sequence,generated:frame.generated,time:drawable.presentedTime)
        }
        // Keep both CVPixelBuffer and texture alive until the last GPU read.
        cb.addCompletedHandler { [frame] command in
            _ = frame
            if command.status == .error { measurements.error() }
            slots.signal()
        }
        cb.commit()
        presented += 1
        if frame.presentationID != lastSequence { uniquePresented += 1; lastSequence = frame.presentationID }
        lastSourceAgeMS = (now-frame.sourceSeconds)*1000
    }
    public override var acceptsFirstResponder:Bool {true}
    public func setMetricsVisible(_ enabled:Bool) {
        statsVisible=enabled;statsHUD.isHidden = !enabled
        UserDefaults.standard.set(enabled,forKey:"DLSS.NativeStatsVisible");needsLayout=true
    }
    public override func keyDown(with event:NSEvent) {if event.keyCode==100 {statsVisible.toggle();statsHUD.isHidden = !statsVisible;UserDefaults.standard.set(statsVisible,forKey:"DLSS.NativeStatsVisible");needsLayout=true}else{super.keyDown(with:event)}}
    public func stop() { link?.invalidate(); link = nil;pipeline.reportPresentation(["active":false]) }
    deinit { link?.invalidate() }
}
