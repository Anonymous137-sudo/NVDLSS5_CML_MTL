import AppKit
import CoreVideo
import Foundation
import NeuralPost
import MLX
import CoreImage
import ImageIO
import UniformTypeIdentifiers

let args = Array(CommandLine.arguments.dropFirst())
let path = args.first(where: { !$0.hasPrefix("--") && $0.hasSuffix(".safetensors") })
let appearancePath = args.first(where:{$0.hasPrefix("--appearance=")}).map { String($0.dropFirst(13)) }
let seconds = args.first(where: { $0.hasPrefix("--seconds=") }).flatMap { Double($0.dropFirst(10)) } ?? 10
let anePath=args.first(where:{$0.hasPrefix("--ane-model=")}).map { String($0.dropFirst(12)) }
let inputPath=args.first(where:{$0.hasPrefix("--input=")}).map { String($0.dropFirst(8)) }
let outputPath=args.first(where:{$0.hasPrefix("--output=")}).map { String($0.dropFirst(9)) }
let pipeline = try PostPipeline(modelURL: path.map { URL(fileURLWithPath: $0) },appearanceURL:appearancePath.map { URL(fileURLWithPath:$0) },aneURL:anePath.map { URL(fileURLWithPath:$0) })
func makeInput(_ phase:Int) -> CVPixelBuffer {
var pixels: CVPixelBuffer?
let attrs: [CFString:Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true]
precondition(CVPixelBufferCreate(nil,1280,720,kCVPixelFormatType_32BGRA,attrs as CFDictionary,&pixels) == kCVReturnSuccess)
let input = pixels!
CVPixelBufferLockBaseAddress(input,[])
let base = CVPixelBufferGetBaseAddress(input)!.assumingMemoryBound(to: UInt8.self)
let stride = CVPixelBufferGetBytesPerRow(input)
for y in 0..<720 { for x in 0..<1280 {
    let i = y*stride+x*4
    base[i] = UInt8((x+phase*32)%256); base[i+1] = UInt8((y+phase*16)%256); base[i+2] = UInt8((x+y+phase*32)%256); base[i+3] = 255
} }
CVPixelBufferUnlockBaseAddress(input,[])
if let inputPath {
    guard let image=CIImage(contentsOf:URL(fileURLWithPath:inputPath)),image.extent.size==CGSize(width:1280,height:720) else { fatalError("Input must be 1280x720") }
    CIContext().render(image,to:input,bounds:image.extent,colorSpace:CGColorSpace(name:CGColorSpace.sRGB))
}
return input
}
let inputs = (0..<(args.contains("--moving") ? 4 : 1)).map(makeInput)
var presenter: NativePresenter?
var window: NSWindow?
if args.contains("--preview") {
    NSApplication.shared.setActivationPolicy(.regular)
    presenter = try NativePresenter(pipeline: pipeline)
    window = NSWindow(contentRect: presenter!.bounds,styleMask: [.titled,.closable,.resizable],backing: .buffered,defer: false)
    window!.title = "OBS Neural Post · 120 Hz request / 60 Hz synthetic input"
    window!.contentView = presenter; window!.center(); window!.makeKeyAndOrderFront(nil)
    NSApplication.shared.activate(ignoringOtherApps: true)
}
let lock = NSLock()
var measurements: [Double] = [], gpu: [Double] = [], ane:[Double] = []
var lastFrame:ProcessedFrame?
var lastSequence: UInt64 = 0
pipeline.onComplete = { frame in
    lock.lock(); defer { lock.unlock() }
    lastFrame=frame
    if frame.aneMS>0 { ane.append(frame.aneMS) }
    // Exclude first 30 completions for model/kernel warmup.
    if lastSequence >= 30 { measurements.append(frame.processingMS); gpu.append(frame.gpuMS) }
    lastSequence += 1
}
let start = ProcessInfo.processInfo.systemUptime
var sequence: UInt64 = 0
while ProcessInfo.processInfo.systemUptime-start < seconds {
    let now = ProcessInfo.processInfo.systemUptime
    let due = UInt64((now-start)*60)+1
    if due > sequence { sequence = due; pipeline.submit(inputs[Int(sequence)%inputs.count],sequence: sequence,sourceSeconds: now) }
    RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.001))
}
pipeline.stop(); presenter?.stop()
while pipeline.stats.busy { RunLoop.main.run(until: Date(timeIntervalSinceNow: 0.01)) }
if presenter != nil { RunLoop.main.run(until:Date(timeIntervalSinceNow:0.1)) }
lock.lock(); let times = measurements.sorted(); let gpuTimes = gpu; lock.unlock()
func percentile(_ p: Double) -> Double { times.isEmpty ? 0 : times[min(times.count-1,Int(Double(times.count-1)*p))] }
let s = pipeline.stats
var report: [String:Any] = ["source":inputs.count>1 ? "synthetic-4-phase-720p" : "synthetic-static-720p", "seconds":seconds, "submitted":s.submitted,
    "completed":s.completed,"neuralCompleted":s.neuralCompleted,"appearanceCompleted":s.appearanceCompleted,"superseded":s.superseded,"failed":s.failed,
    "completionFPS":Double(s.completed)/seconds,"warmSamples":times.count,
    "processingP50MS":percentile(0.5),"processingP95MS":percentile(0.95),"processingP99MS":percentile(0.99),
    "meanMPSAndPackGPUms":gpuTimes.isEmpty ? 0 : gpuTimes.reduce(0,+)/Double(gpuTimes.count),
    "ane":anePath != nil,"error":s.error]
report["source"] = inputPath == nil ? report["source"] : "captured-static-720p-replayed-at-60Hz"
let aneSorted=ane.sorted()
report["anePreparationPredictionTransferP50MS"]=aneSorted.isEmpty ? 0 : aneSorted[aneSorted.count/2]
report["anePreparationPredictionTransferP95MS"]=aneSorted.isEmpty ? 0 : aneSorted[Int(Double(aneSorted.count-1)*0.95)]
if let outputPath,let frame=lastFrame {
    let image=CIImage(cvPixelBuffer:frame.pixels),context=CIContext()
    if let cg=context.createCGImage(image,from:image.extent),let writer=CGImageDestinationCreateWithURL(URL(fileURLWithPath:outputPath) as CFURL,UTType.png.identifier as CFString,1,nil) {
        CGImageDestinationAddImage(writer,cg,nil);report["imageExported"]=CGImageDestinationFinalize(writer)
    }
}
report["mlxActiveMemoryBytes"] = Memory.activeMemory
report["mlxCacheMemoryBytes"] = Memory.cacheMemory
report["mlxPeakMemoryBytes"] = Memory.peakMemory
if let presenter {
    report["displayCallbacks"] = presenter.callbacks; report["displaySubmissions"] = presenter.presented
    report["uniqueDisplaySubmissions"] = presenter.uniquePresented
    report["lastDisplayCallbackIntervalMS"] = presenter.lastCallbackIntervalMS
    report["lastSourceAgeMS"] = presenter.lastSourceAgeMS
    let displayed = presenter.displayStats
    report["drawablePresentedCount"] = displayed.count; report["drawableUniquePresentedCount"] = displayed.unique
    report["drawablePresentationRateHz"] = displayed.rate; report["displayGPUFailures"] = displayed.errors
    report["displayActiveDurationSeconds"] = displayed.lastTime-displayed.firstTime
    report["drawableWidth"] = presenter.drawablePixels.width; report["drawableHeight"] = presenter.drawablePixels.height
}
print(String(data: try JSONSerialization.data(withJSONObject: report,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
if s.failed > 0 || !s.error.isEmpty || s.completed == 0 { exit(1) }
