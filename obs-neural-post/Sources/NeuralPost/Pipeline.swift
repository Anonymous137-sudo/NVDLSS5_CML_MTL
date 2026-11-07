import CoreVideo
import Foundation
import Metal
import MetalPerformanceShaders
import MLX

public final class ProcessedFrame {
    public let pixels: CVPixelBuffer
    public let texture: MTLTexture
    public let sequence: UInt64
    public let sourceSeconds: Double
    public let neural: Bool
    public let appearance: Bool
    public let processingMS: Double
    public let gpuMS: Double
    public let aneMS: Double
    init(_ pixels: CVPixelBuffer, _ texture: MTLTexture, _ sequence: UInt64,
         _ sourceSeconds: Double, _ neural: Bool, _ processingMS: Double, _ gpuMS: Double, appearance: Bool = false, aneMS:Double = 0) {
        self.pixels = pixels; self.texture = texture; self.sequence = sequence
        self.sourceSeconds = sourceSeconds; self.neural = neural
        self.processingMS = processingMS; self.gpuMS = gpuMS
        self.appearance = appearance; self.aneMS = aneMS
    }
}

public struct PipelineStats {
    public var submitted: UInt64 = 0, superseded: UInt64 = 0, completed: UInt64 = 0
    public var neuralCompleted: UInt64 = 0, appearanceCompleted: UInt64 = 0, failed: UInt64 = 0
    public var error: String = ""
    public var pending = 0, busy = false
}

/// One worker, one replaceable pending input, immutable completed output.
/// Capture never waits for inference. All MLX use is serialized on this worker.
public final class PostPipeline {
    public let experimentalAppearance:Bool
    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let decode: MTLComputePipelineState
    private let encode: MTLComputePipelineState
    private let deltaPipeline: MTLComputePipelineState
    private var deltaTexture: MTLTexture?
    private let scaler: MPSImageLanczosScale
    private let worker = DispatchQueue(label: "obs.neural-post.inference", qos: .userInitiated)
    private let lock = NSLock()
    private var pending: (CVPixelBuffer, UInt64, Double, UInt64)?
    private var latestFrame: ProcessedFrame?
    private var state = PipelineStats()
    private var epoch: UInt64 = 0
    private var stopped = false
    private var model: CompactSR?
    private var appearance: CompactAppearance?
    private var ane: ANEAppearance?
    private let noAppearance: MTLBuffer
    private let validation: MTLBuffer
    private var inputStorage: MTLBuffer?
    private var inputStorageSize = 0
    private let modelURL: URL?
    public var onComplete: ((ProcessedFrame) -> Void)?

    public init(modelURL: URL? = nil, appearanceURL: URL? = nil, aneURL:URL? = nil) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw Self.error("Metal unavailable")
        }
        self.device = device; self.queue = queue; self.modelURL = modelURL; experimentalAppearance = aneURL != nil
        guard let zeros = device.makeBuffer(length: 80*45*12*2,options:.storageModeShared) else { throw Self.error("Cannot allocate coefficient buffer") }
        zeros.contents().initializeMemory(as:UInt8.self,repeating:0,count:zeros.length); noAppearance = zeros
        guard let validation = device.makeBuffer(length:4,options:.storageModeShared) else { throw Self.error("Cannot allocate validation flag") }
        self.validation = validation
        if aneURL != nil && (appearanceURL != nil || modelURL == nil) { throw Self.error("ANE pilot requires neural SR and cannot stack the legacy appearance transform") }
        if appearanceURL != nil && modelURL == nil { throw Self.error("Appearance processing currently requires the neural SR model") }
        let lib = try device.makeLibrary(source: Self.shaders, options: nil)
        decode = try device.makeComputePipelineState(function: lib.makeFunction(name: "unpack")!)
        encode = try device.makeComputePipelineState(function: lib.makeFunction(name: "pack")!)
        deltaPipeline = try device.makeComputePipelineState(function:lib.makeFunction(name:"appearance_delta")!)
        scaler = MPSImageLanczosScale(device: device)
        // Model creation happens once on the inference queue, before frame work.
        worker.async { [self] in
            if let modelURL {
                do {
                    model = try CompactSR(url: modelURL)
                    if let appearanceURL { appearance = try CompactAppearance(url: appearanceURL) }
                    if let aneURL { ane = try ANEAppearance(url:aneURL,device:device) }
                }
                catch { model = nil; lock.lock(); state.error = String(describing: error); lock.unlock() }
            }
        }
    }
    public var stats: PipelineStats { lock.lock(); defer { lock.unlock() }; return state }
    public func latest() -> ProcessedFrame? { lock.lock(); defer { lock.unlock() }; return latestFrame }
    public func submit(_ buffer: CVPixelBuffer, sequence: UInt64, sourceSeconds: Double) {
        lock.lock()
        guard !stopped else { lock.unlock(); return }
        state.submitted += 1
        if pending != nil { state.superseded += 1 }
        pending = (buffer, sequence, sourceSeconds, epoch); state.pending = 1
        let start = !state.busy
        if start { state.busy = true }
        lock.unlock()
        if start { worker.async { [self] in drain() } }
    }
    public func reset() {
        lock.lock(); epoch &+= 1; pending = nil; latestFrame = nil; state.pending = 0; lock.unlock()
    }
    public func stop() {
        lock.lock(); stopped = true; epoch &+= 1; pending = nil; latestFrame = nil; state.pending = 0; lock.unlock()
    }
    private func drain() {
        while true {
            lock.lock()
            guard let job = pending else { state.busy = false; lock.unlock(); return }
            pending = nil; state.pending = 0; lock.unlock()
            autoreleasepool {
                do {
                    let result = try process(job.0, sequence: job.1, sourceSeconds: job.2)
                    lock.lock()
                    let publish = !stopped && job.3 == epoch
                    if publish {
                        latestFrame = result; state.completed += 1
                        if result.neural { state.neuralCompleted += 1 }
                        if result.appearance { state.appearanceCompleted += 1 }
                    }
                    lock.unlock()
                    if publish { onComplete?(result) }
                } catch {
                    lock.lock(); state.failed += 1; state.error = String(describing: error); lock.unlock()
                }
            }
        }
    }
    private static func error(_ text: String) -> NSError {
        NSError(domain: "NeuralPost", code: 2, userInfo: [NSLocalizedDescriptionKey: text])
    }
    private func texture(_ buffer: CVPixelBuffer) throws -> MTLTexture {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA,
              let surface = CVPixelBufferGetIOSurface(buffer)?.takeUnretainedValue() else {
            throw Self.error("Expected IOSurface-backed SDR BGRA8 capture")
        }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer), mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d, iosurface: surface, plane: 0) else {
            throw Self.error("Cannot wrap IOSurface")
        }
        return t
    }
    private func dispatch(_ encoder: MTLComputeCommandEncoder, _ p: MTLComputePipelineState, _ w: Int, _ h: Int) {
        encoder.setComputePipelineState(p)
        encoder.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        encoder.endEncoding()
    }
    private func finish(_ cb: MTLCommandBuffer) throws -> Double {
        // Only the private worker waits at the MPS/MLX ownership boundary.
        // Capture, OBS's render thread and the ProMotion display link never wait.
        cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed else { throw cb.error ?? Self.error("GPU command failed") }
        return (cb.gpuEndTime - cb.gpuStartTime) * 1000
    }
    private func process(_ buffer: CVPixelBuffer, sequence: UInt64, sourceSeconds: Double) throws -> ProcessedFrame {
        let start = ProcessInfo.processInfo.systemUptime
        let input = try texture(buffer)
        guard input.width == 1280 && input.height == 720 else { throw Self.error("Select 1280×720 capture; other sizes are not silently resampled") }
        // Fresh output avoids recycling pixels while OBS's external graphics
        // backend still samples an older surface. Add pooling only with a consumer fence.
        var output: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true]
        guard CVPixelBufferCreate(nil, 3840, 2160, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &output) == kCVReturnSuccess,
              let output else { throw Self.error("Cannot allocate output") }
        CVBufferSetAttachment(output, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(output, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_sRGB, .shouldPropagate)
        let target = try texture(output)
        let request=ane?.start(buffer)
        var aneResult:AppearanceFieldResult?
        var gpuMS = 0.0
        if let model {
            let bytes = 1280 * 720 * 3 * 2
            if bytes != inputStorageSize {
                inputStorage = device.makeBuffer(length: bytes, options: .storageModeShared); inputStorageSize = bytes
            }
            guard let storage = inputStorage, let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw Self.error("Cannot prepare model input") }
            enc.setTexture(input, index: 0); enc.setBuffer(storage, offset: 0, index: 0)
            dispatch(enc, decode, 1280, 720); gpuMS += try finish(cb)
            let array = MLXArray(rawPointer: storage.contents(), [1,720,1280,3], dtype: .float16) { _ = storage }
            let result = model.head(array)
            let coefficients = appearance?(array)
            if let coefficients { eval(result,coefficients) } else { eval(result) }
            guard let rgb = result.asMTLBuffer(device:device,noCopy:true), let post = queue.makeCommandBuffer() else { throw Self.error("Cannot bridge model output") }
            aneResult = try request?.take()
            validation.contents().storeBytes(of:UInt32(0),as:UInt32.self)
            if let aneResult {
                if deltaTexture == nil {
                    let d=MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.rgba16Float,width:1280,height:720,mipmapped:false)
                    d.storageMode = .private; d.usage = [.shaderRead,.shaderWrite]
                    deltaTexture=device.makeTexture(descriptor:d)
                }
                guard let deltaTexture,let encoder=post.makeComputeCommandEncoder() else { throw Self.error("Cannot encode appearance delta") }
                encoder.setTexture(input,index:0);encoder.setTexture(deltaTexture,index:1)
                encoder.setBuffer(aneResult.buffer,offset:0,index:0);encoder.setBuffer(validation,offset:0,index:1)
                var mode=aneResult.mode;encoder.setBytes(&mode,length:4,index:2)
                dispatch(encoder,deltaPipeline,1280,720)
            }
            guard let out=post.makeComputeCommandEncoder(),let outputBias=model.outputBias.asMTLBuffer(device:device,noCopy:true) else { throw Self.error("Cannot encode output") }
            out.setBuffer(rgb,offset:0,index:0);out.setTexture(target,index:0)
            out.setTexture(aneResult == nil ? input : deltaTexture,index:1)
            out.setBuffer(outputBias,offset:0,index:3);out.setBuffer(validation,offset:0,index:4)
            let coefficientBuffer = coefficients?.asMTLBuffer(device:device,noCopy:true) ?? noAppearance
            out.setBuffer(aneResult?.buffer ?? coefficientBuffer,offset:0,index:1)
            var appearanceEnabled: UInt32 = aneResult?.mode ?? (coefficients == nil ? 0 : 1)
            out.setBytes(&appearanceEnabled,length:4,index:2)
            dispatch(out, encode, 3840, 2160); gpuMS += try finish(post)
            guard validation.contents().load(as:UInt32.self) == 0 else { throw Self.error("Neural model produced non-finite output; frame rejected") }
            withExtendedLifetime((result,coefficients)) {}
        } else {
            if modelURL != nil { throw Self.error(stats.error.isEmpty ? "Model unavailable" : stats.error) }
            guard let cb = queue.makeCommandBuffer() else { throw Self.error("Cannot create command") }
            scaler.encode(commandBuffer: cb, sourceTexture: input, destinationTexture: target)
            gpuMS += try finish(cb)
        }
        return ProcessedFrame(output, target, sequence, sourceSeconds, model != nil,
                              (ProcessInfo.processInfo.systemUptime-start)*1000, gpuMS, appearance:appearance != nil || aneResult != nil, aneMS:aneResult?.milliseconds ?? 0)
    }
    private static let shaders = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void unpack(texture2d<float,access::read> src [[texture(0)]], device half *dst [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
        if(p.x>=src.get_width()||p.y>=src.get_height()) return;
        float3 c=src.read(p).rgb; uint i=(p.y*src.get_width()+p.x)*3;
        dst[i]=half(c.r); dst[i+1]=half(c.g); dst[i+2]=half(c.b);
    }
    float3 apply_ane(float3 color,const device half *coeff,float2 uv,device atomic_uint *invalid,float3 highpass,uint mode) {
            float2 q=clamp(uv*float2(192,108)-.5f,float2(0),float2(191,107));
            uint2 a=uint2(q), b=min(a+1,uint2(191,107)); float2 f=fract(q);
            float fields[16];
            for(uint k=0;k<16;k++) {
                uint base=k*108*192;
                fields[k]=mix(mix(float(coeff[base+a.y*192+a.x]),float(coeff[base+a.y*192+b.x]),f.x),
                              mix(float(coeff[base+b.y*192+a.x]),float(coeff[base+b.y*192+b.x]),f.x),f.y);
                if(!isfinite(fields[k])) atomic_store_explicit(invalid,1u,memory_order_relaxed);
            }
            // Same conservative GTA minimap reservation used by pilot training.
            bool protectedUI=uv.x<.26f && uv.y>=.70f;
            if(!protectedUI) {
                color=clamp(color,0.f,1.f);
                float3 edit;
                float confidence=1.f/(1.f+exp(-clamp(fields[15],-16.f,16.f)));
                for(uint ch=0;ch<3;ch++) {
                    float4 c;
                    for(uint k=0;k<4;k++) c[k]=.125f*tanh(fields[ch*4+k]);
                    float detail=mode==3 ? 2.f*tanh(fields[12+ch])*highpass[ch] : .08f*tanh(fields[12+ch]);
                    edit[ch]=confidence*(dot(c,float4(color,1))+detail);
                }
                float amount=1.f;
                for(uint ch=0;ch<3;ch++) {
                    float headroom=edit[ch]>=0.f ? 1.f-color[ch] : color[ch];
                    amount=min(amount,headroom/(abs(edit[ch])+1e-6f));
                }
                color+=clamp(amount,0.f,1.f)*edit;
            }
        return color;
    }
    kernel void appearance_delta(texture2d<float,access::read> src [[texture(0)]],texture2d<float,access::write> dst [[texture(1)]],const device half *coeff [[buffer(0)]],device atomic_uint *invalid [[buffer(1)]],constant uint &mode [[buffer(2)]],uint2 p [[thread_position_in_grid]]) {
        if(p.x>=src.get_width()||p.y>=src.get_height()) return;
        float3 color=src.read(p).rgb;
        float2 uv=(float2(p)+.5f)/float2(src.get_width(),src.get_height());
        float3 local=0;
        if(mode==3) {
            for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
                int2 n=clamp(int2(p)+int2(dx,dy),int2(0),int2(src.get_width()-1,src.get_height()-1));
                local+=src.read(uint2(n)).rgb/9.f;
            }
        }
        dst.write(float4(apply_ane(color,coeff,uv,invalid,color-local,mode)-color,0),p);
    }
    kernel void pack(const device half *src [[buffer(0)]], const device half *coeff [[buffer(1)]], constant uint &appearance [[buffer(2)]], const device half *bias [[buffer(3)]], device atomic_uint *invalid [[buffer(4)]], texture2d<float,access::write> dst [[texture(0)]], texture2d<float> delta [[texture(1)]], uint2 p [[thread_position_in_grid]]) {
        if(p.x>=dst.get_width()||p.y>=dst.get_height()) return;
        uint i=((p.y/6)*(dst.get_width()/6)+p.x/6)*108+(p.y%6)*6+p.x%6;
        uint bidx=(p.y%6)*6+p.x%6;
        float3 color=float3(src[i]+bias[bidx],src[i+36]+bias[bidx+36],src[i+72]+bias[bidx+72]);
        if(appearance==1) {
            float2 q=clamp((float2(p)+.5f)*float2(80.f/3840.f,45.f/2160.f)-.5f,float2(0),float2(79,44));
            uint2 a=uint2(q), b=min(a+1,uint2(79,44)); float2 f=fract(q);
            float3 edit;
            for(uint ch=0;ch<3;ch++) {
                float4 c;
                for(uint k=0;k<4;k++) {
                    uint offset=ch*4+k;
                    float v0=mix(float(coeff[(a.y*80+a.x)*12+offset]),float(coeff[(a.y*80+b.x)*12+offset]),f.x);
                    float v1=mix(float(coeff[(b.y*80+a.x)*12+offset]),float(coeff[(b.y*80+b.x)*12+offset]),f.x);
                    c[k]=mix(v0,v1,f.y);
                }
                edit[ch]=dot(c,float4(color,1));
            }
            // Bound edit amplitude. Scale RGB together near gamut boundaries.
            edit=clamp(edit,-.25f,.25f); float amount=1.f;
            for(uint ch=0;ch<3;ch++) {
                if(edit[ch]>0) amount=min(amount,(1.f-color[ch])/edit[ch]);
                if(edit[ch]<0) amount=min(amount,-color[ch]/edit[ch]);
            }
            color+=clamp(amount,0.f,1.f)*edit;
        }
        if(appearance>=2) {
            float2 uv=(float2(p)+.5f)/float2(dst.get_width(),dst.get_height());
            if(!(uv.x<.26f && uv.y>=.70f)) {
                constexpr sampler linearSampler(filter::linear,address::clamp_to_edge);
                color+=delta.sample(linearSampler,uv).rgb;
            }
        }
        if(!all(isfinite(color))) atomic_store_explicit(invalid,1u,memory_order_relaxed);
        dst.write(float4(clamp(color,0.f,1.f),1),p);
    }
    """
}
