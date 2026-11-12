import AppKit
import CoreVideo
import DLSSNR

@_cdecl("nr_abi_version") public func nrABIVersion()->UInt32 { 1 }

private final class NRBridge {
 let pipeline:PostPipeline
 var window:NSWindow?
 var filterWindow:NSWindow?
 init(_ config:NRConfiguration)throws {pipeline=try PostPipeline(config)}
}
private func config(_ text:UnsafePointer<CChar>)throws->NRConfiguration {try JSONDecoder().decode(NRConfiguration.self,from:Data(String(cString:text).utf8)).checked()}
@_cdecl("nr_set_metallib") public func nrSetMetallib(_ path:UnsafePointer<CChar>) {_ = path /* MLX 0.31 uses the packaged colocated library. */}
@_cdecl("nr_create") public func nrCreate(_ text:UnsafePointer<CChar>)->UnsafeMutableRawPointer? {
 do {return Unmanaged.passRetained(try NRBridge(config(text))).toOpaque()}
 catch {fputs("DLSS NR: \(error)\n",stderr);return nil}
}
@_cdecl("nr_update") public func nrUpdate(_ handle:UnsafeMutableRawPointer?,_ text:UnsafePointer<CChar>)->Bool {
 guard let handle else{return false}
 do {Unmanaged<NRBridge>.fromOpaque(handle).takeUnretainedValue().pipeline.update(try config(text));return true}
 catch {fputs("DLSS NR settings: \(error)\n",stderr);return false}
}
@_cdecl("nr_reset_input") public func nrResetInput(_ handle:UnsafeMutableRawPointer?) {
 guard let handle else{return}
 Unmanaged<NRBridge>.fromOpaque(handle).takeUnretainedValue().pipeline.resetInput()
}
@_cdecl("nr_destroy") public func nrDestroy(_ handle:UnsafeMutableRawPointer?) {
 guard let handle else{return};let bridge=Unmanaged<NRBridge>.fromOpaque(handle).takeRetainedValue();bridge.pipeline.stop()
 DispatchQueue.main.async {bridge.window?.close();bridge.window=nil;bridge.filterWindow?.close();bridge.filterWindow=nil}
}
@_cdecl("nr_submit") public func nrSubmit(_ handle:UnsafeMutableRawPointer?,_ buffer:UnsafeMutableRawPointer?,_ sequence:UInt64,_ pts:Double) {
 guard let handle,let buffer else{return}
 Unmanaged<NRBridge>.fromOpaque(handle).takeUnretainedValue().pipeline.submit(Unmanaged<CVPixelBuffer>.fromOpaque(buffer).takeUnretainedValue(),sequence:sequence,sourceSeconds:pts)
}
@_cdecl("nr_frame_acquire") public func nrFrameAcquire(_ handle:UnsafeMutableRawPointer?)->UnsafeMutableRawPointer? {
 guard let handle,let frame=Unmanaged<NRBridge>.fromOpaque(handle).takeUnretainedValue().pipeline.latest() else{return nil}
 return Unmanaged.passRetained(frame).toOpaque()
}
@_cdecl("nr_frame_pixels") public func nrFramePixels(_ frame:UnsafeMutableRawPointer)->UnsafeMutableRawPointer {Unmanaged.passUnretained(Unmanaged<ProcessedFrame>.fromOpaque(frame).takeUnretainedValue().pixels).toOpaque()}
@_cdecl("nr_frame_sequence") public func nrFrameSequence(_ frame:UnsafeMutableRawPointer)->UInt64 {Unmanaged<ProcessedFrame>.fromOpaque(frame).takeUnretainedValue().presentationID}
@_cdecl("nr_frame_kind") public func nrFrameKind(_ frame:UnsafeMutableRawPointer)->Int32 {Unmanaged<ProcessedFrame>.fromOpaque(frame).takeUnretainedValue().captureGuided ? 2 : Unmanaged<ProcessedFrame>.fromOpaque(frame).takeUnretainedValue().generated ? 1 : 0}
@_cdecl("nr_frame_release") public func nrFrameRelease(_ frame:UnsafeMutableRawPointer?) {if let frame {Unmanaged<ProcessedFrame>.fromOpaque(frame).release()}}
@_cdecl("nr_handle_retain") public func nrHandleRetain(_ handle:UnsafeMutableRawPointer?)->UnsafeMutableRawPointer? {
 guard let handle else{return nil};_ = Unmanaged<NRBridge>.fromOpaque(handle).retain();return handle
}
@_cdecl("nr_handle_release") public func nrHandleRelease(_ handle:UnsafeMutableRawPointer?) {if let handle {Unmanaged<NRBridge>.fromOpaque(handle).release()}}
@_cdecl("nr_set_status_overlay") public func nrSetStatusOverlay(_ handle:UnsafeMutableRawPointer?,_ enabled:Bool) {guard let handle else{return};Unmanaged<NRBridge>.fromOpaque(handle).takeUnretainedValue().pipeline.setStatusOverlay(enabled)}
@_cdecl("nr_status") public func nrStatus(_ handle:UnsafeMutableRawPointer?,_ output:UnsafeMutablePointer<CChar>,_ count:Int32) {
 guard let handle,count>0 else{return}
 let report=Unmanaged<NRBridge>.fromOpaque(handle).takeUnretainedValue().pipeline.status()
 let status: [String:Any]=JSONSerialization.isValidJSONObject(report) ? report : ["error":"Non-finite runtime telemetry rejected","mfgError":"Invalid telemetry"]
 if let data=try? JSONSerialization.data(withJSONObject:status,options:[.sortedKeys]),let text=String(data:data,encoding:.utf8) {text.withCString{_ = strlcpy(output,$0,Int(count))}}
}
@_cdecl("nr_show_preview") public func nrShowPreview(_ handle:UnsafeMutableRawPointer?) {
 guard let handle else{return};let bridge=Unmanaged<NRBridge>.fromOpaque(handle).takeUnretainedValue()
 DispatchQueue.main.async {
  if bridge.window==nil {
   do {
    let view=try NativePresenter(pipeline:bridge.pipeline)
    let window=NSWindow(contentRect:view.bounds,styleMask:[.titled,.closable,.resizable,.miniaturizable],backing:.buffered,defer:false)
    window.isReleasedWhenClosed=false;window.contentView=view;window.center();bridge.window=window
   } catch {fputs("DLSS NR preview: \(error)\n",stderr)}
  }
  NSApplication.shared.activate()
  bridge.window?.makeKeyAndOrderFront(nil)
 }
}

@_cdecl("nr_hide_output") public func nrHideOutput(_ handle:UnsafeMutableRawPointer?) {
 guard let handle else{return};let bridge=Unmanaged<NRBridge>.fromOpaque(handle).takeUnretainedValue()
 DispatchQueue.main.async {bridge.window?.orderOut(nil);bridge.filterWindow?.orderOut(nil)}
}
@_cdecl("nr_show_screen_filter") public func nrShowScreenFilter(_ handle:UnsafeMutableRawPointer?,_ display:UInt32) {
 guard let handle else{return};let bridge=Unmanaged<NRBridge>.fromOpaque(handle).takeUnretainedValue()
 DispatchQueue.main.async {
  guard let screen=NSScreen.screens.first(where:{($0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value==display}) else{return}
  bridge.filterWindow?.close();bridge.filterWindow=nil;bridge.window?.orderOut(nil)
  do {
   let view=try NativePresenter(pipeline:bridge.pipeline)
   let window=NSWindow(contentRect:screen.frame,styleMask:.borderless,backing:.buffered,defer:false)
   window.isReleasedWhenClosed=false;window.contentView=view;window.ignoresMouseEvents=true
   window.level = .floating;window.collectionBehavior=[.canJoinAllSpaces,.fullScreenAuxiliary,.stationary]
   window.backgroundColor = .black;window.isOpaque=true;window.setFrame(screen.frame,display:true)
   bridge.filterWindow=window;window.orderFrontRegardless()
  } catch {fputs("DLSS screen filter: \(error)\n",stderr)}
 }
}

@_cdecl("nr_set_metrics_overlay") public func nrSetMetricsOverlay(_ handle:UnsafeMutableRawPointer?,_ enabled:Bool) {
 guard let handle else{return};let bridge=Unmanaged<NRBridge>.fromOpaque(handle).takeUnretainedValue()
 DispatchQueue.main.async {
  UserDefaults.standard.set(enabled,forKey:"DLSS.NativeStatsVisible")
  (bridge.window?.contentView as? NativePresenter)?.setMetricsVisible(enabled)
  (bridge.filterWindow?.contentView as? NativePresenter)?.setMetricsVisible(enabled)
 }
}
