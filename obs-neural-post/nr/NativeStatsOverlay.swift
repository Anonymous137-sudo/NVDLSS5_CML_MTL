import AppKit
import Darwin
import IOKit

public struct DisplayPresentationStats {
    public var count:UInt64 = 0, unique:UInt64 = 0
    public var firstTime:Double = 0, lastTime:Double = 0
    public var lastSequence:UInt64 = 0
    public var errors:UInt64 = 0
    public var freshShownFPS:Double = 0
    public var uniqueIntervalP50MS:Double=0,uniqueIntervalP95MS:Double=0,uniqueIntervalP99MS:Double=0,uniqueIntervalMaxMS:Double=0
    public var frameIntervals:[Double]=[]
    public var holdsOver33MS=0
    public var baseShownFPS:Double=0,generatedShownFPS:Double=0
    public var baseShown:UInt64=0,generatedShown:UInt64=0,repeated:UInt64=0
    public var measuredHz:Double=0
    public var rate:Double {measuredHz}
}
package final class DisplayMeasurements {
    package init() {}
    let lock = NSLock()
    var stats = DisplayPresentationStats()
    var recentUniqueTimes:[Double] = []
    var intervals:[Double]=[],presentTimes:[Double]=[]
    package func resetWindow() {lock.lock();recentUniqueTimes=[];recentBaseTimes=[];recentGeneratedTimes=[];intervals=[];presentTimes=[];lock.unlock()}
    var recentBaseTimes:[Double]=[],recentGeneratedTimes:[Double]=[]
    var basePresented:UInt64=0,generatedPresented:UInt64=0,repeatedPresented:UInt64=0
    package func presented(sequence:UInt64,generated:Bool,time:Double) {
        guard time>0 else { return }
        lock.lock(); defer { lock.unlock() }
        if stats.count==0 { stats.firstTime=time }
        stats.count += 1; stats.lastTime=time
        presentTimes.append(time);if presentTimes.count>256 {presentTimes.removeFirst(presentTimes.count-256)}
        if sequence != stats.lastSequence {
            stats.unique += 1; stats.lastSequence=sequence
            if generated {generatedPresented += 1;recentGeneratedTimes.append(time)} else {basePresented += 1;recentBaseTimes.append(time)}
            if recentGeneratedTimes.count>256 {recentGeneratedTimes.removeFirst(recentGeneratedTimes.count-256)}
            if recentBaseTimes.count>256 {recentBaseTimes.removeFirst(recentBaseTimes.count-256)}
            if let last=recentUniqueTimes.last,time>last {intervals.append((time-last)*1000);if intervals.count>256 {intervals.removeFirst(intervals.count-256)}}
            recentUniqueTimes.append(time)
            if recentUniqueTimes.count>256 { recentUniqueTimes.removeFirst(recentUniqueTimes.count-256) }
        } else {repeatedPresented += 1}
    }
    package func error() { lock.lock(); stats.errors += 1; lock.unlock() }
    package func snapshot() -> DisplayPresentationStats {
        lock.lock(); defer { lock.unlock() }
        var result=stats
        let now=ProcessInfo.processInfo.systemUptime
        result.freshShownFPS=Double(recentUniqueTimes.filter{$0>now-2 && $0<=now}.count)/2
        result.baseShownFPS=Double(recentBaseTimes.filter{$0>now-2 && $0<=now}.count)/2
        result.generatedShownFPS=Double(recentGeneratedTimes.filter{$0>now-2 && $0<=now}.count)/2
        if let first=presentTimes.first,let last=presentTimes.last,last>first,now-last<2 {result.measuredHz=Double(presentTimes.count-1)/(last-first)}
        let ordered=intervals.sorted()
        func percentile(_ q:Double)->Double {ordered.isEmpty ? 0 : ordered[min(ordered.count-1,Int(Double(ordered.count-1)*q))]}
        result.frameIntervals=intervals
        result.uniqueIntervalP50MS=percentile(0.5);result.uniqueIntervalP95MS=percentile(0.95);result.uniqueIntervalP99MS=percentile(0.99);result.uniqueIntervalMaxMS=ordered.last ?? 0;result.holdsOver33MS=intervals.filter{$0>33.34}.count
        result.baseShown=basePresented;result.generatedShown=generatedPresented;result.repeated=repeatedPresented
        return result
    }
}

private struct HardwareSample {
 var cpu:Double?,gpu:Double?,gpuGB:Double?,appGB:Double?
 var thermal="—"
}
private final class PreviewHardwareTelemetry {
 private let lock=NSLock(),queue=DispatchQueue(label:"obs.preview.hardware",qos:.utility)
 private var sample=HardwareSample(),busy=false,last=0.0,previousCPU:[UInt32]?
 private var accelerator:io_service_t=0
 deinit{if accelerator != 0{IOObjectRelease(accelerator)}}
 func snapshot()->HardwareSample {
  let now=ProcessInfo.processInfo.systemUptime;lock.lock();let value=sample
  if !busy && now-last>=1 {busy=true;last=now;queue.async{[self] in read()}}
  lock.unlock();return value
 }
 private func read(){
  var next=HardwareSample()
  var cpu=host_cpu_load_info(),count=mach_msg_type_number_t(MemoryLayout<host_cpu_load_info>.size/MemoryLayout<integer_t>.size)
  let status=withUnsafeMutablePointer(to:&cpu){$0.withMemoryRebound(to:integer_t.self,capacity:Int(count)){host_statistics(mach_host_self(),HOST_CPU_LOAD_INFO,$0,&count)}}
  if status==KERN_SUCCESS {
   let ticks=[cpu.cpu_ticks.0,cpu.cpu_ticks.1,cpu.cpu_ticks.2,cpu.cpu_ticks.3]
   if let old=previousCPU {let delta=zip(ticks,old).map{Double($0 &- $1)},total=delta.reduce(0,+);if total>0{next.cpu=(1-delta[2]/total)*100}}
   previousCPU=ticks
  }
  var vm=task_vm_info_data_t();var vmCount=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
  let result=withUnsafeMutablePointer(to:&vm){$0.withMemoryRebound(to:integer_t.self,capacity:Int(vmCount)){task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&vmCount)}}
  if result==KERN_SUCCESS{next.appGB=Double(vm.phys_footprint)/1073741824}
  if accelerator==0{accelerator=IOServiceGetMatchingService(kIOMainPortDefault,IOServiceMatching("AGXAccelerator"))}
  if accelerator != 0,let data=IORegistryEntryCreateCFProperty(accelerator,"PerformanceStatistics" as CFString,kCFAllocatorDefault,0)?.takeRetainedValue() as? [String:Any] {
   if let n=data["Device Utilization %"] as? NSNumber{next.gpu=n.doubleValue}
   if let n=data["In use system memory"] as? NSNumber{next.gpuGB=n.doubleValue/1073741824}
  }
  switch ProcessInfo.processInfo.thermalState{case .nominal:next.thermal="NOMINAL";case .fair:next.thermal="FAIR";case .serious:next.thermal="SERIOUS";case .critical:next.thermal="CRITICAL";@unknown default:next.thermal="UNKNOWN"}
  lock.lock();sample=next;busy=false;lock.unlock()
 }
}

/// RTSS-inspired condensed monospace, outlined text and cyan/amber telemetry.
/// Presentation only: never participates in capture, NR or FG image processing.
package final class NativeStatsOverlay:NSView {
 private struct Row{let label:String,value:String,extra:String}
 private var rows:[Row]=[],intervals:[Double]=[]
 private let hardware=PreviewHardwareTelemetry()
 private let cyan=NSColor(calibratedRed:0.05,green:0.85,blue:0.78,alpha:1),amber=NSColor(calibratedRed:1,green:0.69,blue:0.14,alpha:1)
 package override var isFlipped:Bool{true}
 package override func hitTest(_ point:NSPoint)->NSView?{nil}
 package override init(frame:NSRect){super.init(frame:frame);wantsLayer=true;layer?.backgroundColor=NSColor.clear.cgColor;toolTip="RTSS-inspired native telemetry. FPS counts distinct presented frame IDs, not game simulation frames. GPU utilization and unified GPU memory are driver reports; CPU is system load; APP RAM is this process footprint. Temperatures are not fabricated. F8 toggles this overlay."}
 package required init?(coder:NSCoder){fatalError("init(frame:)")}
 package func update(runtime:[String:Any],display:DisplayPresentationStats,sourceAge:Double){
  let hw=hardware.snapshot(),mode=(runtime["mode"] as? String ?? "DLSS").replacingOccurrences(of:"DLSS ",with:"").uppercased(),backend=(runtime["mfgBackend"] as? String ?? "").uppercased()
  let recent=runtime["recentFrames"] as? [[String:Any]] ?? [],fg=recent.suffix(30).compactMap{$0["mfgMS"] as? Double}.filter{$0>0}.sorted()
  let fgMS=fg.isEmpty ? 0:fg[fg.count/2],nr=runtime["processedFPS"] as? Double ?? 0,nrMS=runtime["processingMedianMS"] as? Double ?? 0
  func metric(_ v:Double?,_ suffix:String)->String{v.map{String(format:"%.0f %@",$0,suffix)} ?? "—"}
  let frameMS=display.freshShownFPS>0 ? 1000/display.freshShownFPS:0
  rows=[
   Row(label:"METAL",value:String(format:"%.0f FPS",display.freshShownFPS),extra:String(format:"%.1f ms",frameMS)),
   Row(label:"GPU USAGE",value:metric(hw.gpu,"%"),extra:hw.gpuGB.map{String(format:"%.2f GB",$0)} ?? ""),
   Row(label:"CPU USAGE",value:metric(hw.cpu,"%"),extra:""),
   Row(label:"APP RAM",value:hw.appGB.map{String(format:"%.2f GB",$0)} ?? "—",extra:""),
   Row(label:"NR",value:String(format:"%.1f FPS",nr),extra:String(format:"%.1f ms",nrMS)),
   Row(label:"MFG SHOWN",value:String(format:"%.1f FPS",display.generatedShownFPS),extra:String(format:"%.1f ms",fgMS)),
   Row(label:"DISPLAY",value:String(format:"%.0f Hz",display.measuredHz),extra:""),
   Row(label:"SRC AGE",value:String(format:"%.0f ms",sourceAge),extra:""),
   Row(label:"P95 FRAME",value:String(format:"%.1f ms",display.uniqueIntervalP95MS),extra:""),
   Row(label:"THERM",value:hw.thermal,extra:""),
   Row(label:mode,value:backend,extra:"\(runtime["mfgFactor"] ?? 1)×")]
  intervals=Array(display.frameIntervals.suffix(120));needsDisplay=true
 }
 private func text(_ string:String,_ point:NSPoint,_ color:NSColor,_ size:CGFloat=13.5){
  let font=NSFont(name:"Menlo-Bold",size:size) ?? NSFont.monospacedSystemFont(ofSize:size,weight:.bold)
  let shadow=NSShadow();shadow.shadowColor=NSColor.black;shadow.shadowOffset=NSSize(width:1,height:1);shadow.shadowBlurRadius=1
  let attributes:[NSAttributedString.Key:Any]=[.font:font,.foregroundColor:color,.strokeColor:NSColor.black,.strokeWidth:-2.5,.shadow:shadow]
  NSGraphicsContext.saveGraphicsState();NSGraphicsContext.current?.cgContext.scaleBy(x:0.83,y:1);string.draw(at:NSPoint(x:point.x/0.83,y:point.y),withAttributes:attributes);NSGraphicsContext.restoreGraphicsState()
 }
 package override func draw(_ dirtyRect:NSRect){
  super.draw(dirtyRect)
  for (i,row) in rows.enumerated(){let y=CGFloat(i)*18;text(row.label,NSPoint(x:0,y:y),i==0 ? .white:cyan);text(row.value,NSPoint(x:112,y:y),i==0 ? .white:amber);text(row.extra,NSPoint(x:205,y:y),i==0 ? .white:amber)}
  guard intervals.count>1 else{return}
  let box=NSRect(x:0,y:CGFloat(rows.count)*18+5,width:282,height:31)
  NSColor.black.withAlphaComponent(0.45).setFill();NSBezierPath(rect:box).fill()
  let maximum=max(33.34,intervals.max() ?? 33.34)
  let path=NSBezierPath();for (i,value) in intervals.enumerated(){let p=NSPoint(x:box.minX+CGFloat(i)/CGFloat(intervals.count-1)*box.width,y:box.maxY-CGFloat(min(maximum,max(0,value)))/maximum*box.height);if i==0{path.move(to:p)}else{path.line(to:p)}};cyan.setStroke();path.lineWidth=1;path.stroke()
 }
}
