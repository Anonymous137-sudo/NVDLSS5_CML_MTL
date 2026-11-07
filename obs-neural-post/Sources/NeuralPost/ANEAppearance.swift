import CoreML
import CoreImage
import CoreVideo
import Foundation
import Metal

final class AppearanceFieldResult {
    let buffer:MTLBuffer
    let milliseconds:Double
    let mode:UInt32
    init(buffer:MTLBuffer,milliseconds:Double,mode:UInt32) { self.buffer=buffer; self.milliseconds=milliseconds;self.mode=mode }
}
final class AppearanceRequest {
    private let ready=DispatchSemaphore(value:0)
    private var result:Result<AppearanceFieldResult,Error>?
    func finish(_ value:Result<AppearanceFieldResult,Error>) { result=value; ready.signal() }
    // Only the private processing worker may wait. Capture/render never wait.
    func take() throws -> AppearanceFieldResult {
        guard ready.wait(timeout:.now()+1) == .success else { throw NSError(domain:"NeuralPost",code:25,userInfo:[NSLocalizedDescriptionKey:"ANE request exceeded one-second failure limit"]) }
        return try result!.get()
    }
}
/// Serialized same-frame inference. No latest-result cache and no stale-field reuse.
final class ANEAppearance {
    private let admission=NSLock()
    private var busy=false
    private let mode:UInt32
    private let model:MLModel
    private let queue=DispatchQueue(label:"obs.neural-post.ane",qos:.userInitiated)
    private let context:CIContext
    private let prepared:CVPixelBuffer
    private let device:MTLDevice
    private let color=CGColorSpace(name:CGColorSpace.sRGB)!
    init(url:URL,device:MTLDevice) throws {
        self.device=device; context=CIContext(mtlDevice:device,options:[.cacheIntermediates:false])
        let compiled:URL
        if url.pathExtension == "mlmodelc" { compiled=url }
        else { compiled=try MLModel.compileModel(at:url) }
        defer { if compiled != url { try? FileManager.default.removeItem(at:compiled) } }
        let config=MLModelConfiguration(); config.computeUnits = .cpuAndNeuralEngine
        model=try MLModel(contentsOf:compiled,configuration:config)
        let metadata=model.modelDescription.metadata[.creatorDefinedKey] as? [String:String] ?? [:]
        guard ["neural-post.appearance-fields.v1","neural-post.appearance-fields.v2"].contains(metadata["neural_post_contract"] ?? ""),
              ["pilot-teacher-distilled","checkpoint-unvalidated"].contains(metadata["training_status"] ?? ""),
              let input=model.modelDescription.inputDescriptionsByName["rgb"]?.imageConstraint,
              input.pixelsWide==384,input.pixelsHigh==216,
              model.modelDescription.outputDescriptionsByName["appearance_fields"]?.multiArrayConstraint?.shape.map(\.intValue)==[1,16,108,192] else {
            throw NSError(domain:"NeuralPost",code:20,userInfo:[NSLocalizedDescriptionKey:"Expected teacher-trained appearance-fields v1; random/fixture probes are rejected"])
        }
        mode=metadata["neural_post_contract"] == "neural-post.appearance-fields.v2" ? 3 : 2
        var pixels:CVPixelBuffer?
        let attrs:[CFString:Any]=[kCVPixelBufferIOSurfacePropertiesKey:[:],kCVPixelBufferMetalCompatibilityKey:true]
        guard CVPixelBufferCreate(nil,384,216,kCVPixelFormatType_32BGRA,attrs as CFDictionary,&pixels)==kCVReturnSuccess,let pixels else {
            throw NSError(domain:"NeuralPost",code:21)
        }
        prepared=pixels
    }
    func start(_ source:CVPixelBuffer)->AppearanceRequest {
        let request=AppearanceRequest()
        admission.lock()
        guard !busy else {
            admission.unlock();request.finish(.failure(NSError(domain:"NeuralPost",code:26,userInfo:[NSLocalizedDescriptionKey:"ANE request already in flight"])));return request
        }
        busy=true;admission.unlock()
        queue.async { [self,source] in
            autoreleasepool {
                let value=Result { try predict(source) }
                admission.lock();busy=false;admission.unlock()
                request.finish(value)
            }
        }
        return request
    }
    private func predict(_ source:CVPixelBuffer) throws -> AppearanceFieldResult {
        let began=ProcessInfo.processInfo.systemUptime
        let input=CIImage(cvPixelBuffer:source)
        guard input.extent.width*9==input.extent.height*16 else { throw NSError(domain:"NeuralPost",code:22) }
        let resized=input.transformed(by:CGAffineTransform(scaleX:384/input.extent.width,y:216/input.extent.height))
        context.render(resized,to:prepared,bounds:CGRect(x:0,y:0,width:384,height:216),colorSpace:color)
        let provider=try MLDictionaryFeatureProvider(dictionary:["rgb":MLFeatureValue(pixelBuffer:prepared)])
        let output=try model.prediction(from:provider)
        guard let fields=output.featureValue(for:"appearance_fields")?.multiArrayValue,
              fields.shape.map(\.intValue)==[1,16,108,192],fields.dataType == .float16,
              let buffer=device.makeBuffer(length:16*108*192*2,options:.storageModeShared) else { throw NSError(domain:"NeuralPost",code:23) }
        let strides=fields.strides.map(\.intValue)
        let source=fields.dataPointer.assumingMemoryBound(to:Float16.self)
        let target=buffer.contents().assumingMemoryBound(to:Float16.self)
        for c in 0..<16 { for y in 0..<108 { for x in 0..<192 {
            let v=source[c*strides[1]+y*strides[2]+x*strides[3]]
            guard v.isFinite else { throw NSError(domain:"NeuralPost",code:24,userInfo:[NSLocalizedDescriptionKey:"Nonfinite ANE fields rejected"])}
            target[(c*108+y)*192+x]=v
        } } }
        return AppearanceFieldResult(buffer:buffer,milliseconds:(ProcessInfo.processInfo.systemUptime-began)*1000,mode:mode)
    }
}
