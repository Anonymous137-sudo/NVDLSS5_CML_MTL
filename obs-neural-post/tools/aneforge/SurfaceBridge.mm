// Reuse ANEForge's MIT-licensed compiler/dispatch implementation without changing
// the checkout. This extension binds caller-owned surfaces before first encode.
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include "ANEForgeBridge.h"
#include <chrono>
#include <vector>
#include <cmath>
#include ANEFORGE_DISPATCH_SOURCE

using CreateSurfaceBuffer = int64_t (*)(void **, IOSurfaceRef);
using GetSurface = int64_t (*)(void *, IOSurfaceRef *);

extern "C" int dlss_af_bind_surface(void *handle, const char *name,
                                     int input, IOSurfaceRef surface) {
  auto *program=(ane_e5rt_program_t *)handle;
  if (!program || !name || !surface || program->finalized || program->device_mask != 4 ||
      program->n_ops != 1) return -1;
  auto &op = program->ops[0];
  ane_e5rt_port_t *port = nullptr;
  auto *ports = input ? op.inputs : op.outputs;
  size_t count = input ? op.n_inputs : op.n_outputs;
  for (size_t i = 0; i < count; ++i) if (!strcmp(ports[i].name, name)) port = &ports[i];
  if (!port || IOSurfaceGetAllocSize(surface) < port->size_bytes) return -2;
  auto create = (CreateSurfaceBuffer)dlsym(fw, "e5rt_buffer_object_create_from_iosurface");
  auto get = (GetSurface)dlsym(fw, "e5rt_buffer_object_get_iosurface");
  if (!create || !get) return -3;
  void *buffer = nullptr;
  int64_t rc = create(&buffer, surface);
  if (rc || !buffer) return -4;
  IOSurfaceRef mapped = nullptr;
  rc = get(buffer, &mapped);
  void *address = nullptr;
  if (rc || !mapped || IOSurfaceGetID(mapped) != IOSurfaceGetID(surface) ||
      e5rt_buffer_object_get_data_ptr(buffer, &address) || address != IOSurfaceGetBaseAddress(surface)) {
    e5rt_buffer_object_release(&buffer); return -5;
  }
  if (e5rt_io_port_bind_buffer_object(port->port, buffer)) {
    e5rt_buffer_object_release(&buffer); return -6;
  }
  if (port->buffer) e5rt_buffer_object_release(&port->buffer);
  port->buffer = buffer;
  port->data_ptr = address;
  return 0;
}

static IOSurfaceRef makeSurface(size_t channels, size_t spatial) {
  if (!channels || !spatial || spatial % 32) return nullptr;
  NSDictionary *properties = @{
    (id)kIOSurfaceWidth: @(spatial), (id)kIOSurfaceHeight: @(channels),
    (id)kIOSurfaceBytesPerElement: @2, (id)kIOSurfaceBytesPerRow: @(spatial * 2),
    (id)kIOSurfaceAllocSize: @(channels * spatial * 2)
  };
  return IOSurfaceCreate((__bridge CFDictionaryRef)properties);
}

// Hardware interoperability probe, not a replacement for the NR network.
// Each handle is an independent ANE program for relu(input*0.5 + 0.25).
// Input/output ports are NCHW [1,C,1,S], surfaced as R16Float [S,C].
extern "C" int dlss_af_probe(void **handles, int slots, const char *inputName,
                              const char *outputName, int channels, int spatial,
                              int iterations, const char *reportPath) {
  @autoreleasepool {
    if (!handles || slots < 1 || slots > 3 || channels < 1 || spatial < 32 ||
        spatial % 32 || iterations < slots || !reportPath) return -10;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> gpu = [device newCommandQueue];
    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void fill(texture2d<half,access::write> out [[texture(0)]],constant uint &frame [[buffer(0)]],uint2 p [[thread_position_in_grid]]) {"
      "if(p.x>=out.get_width()||p.y>=out.get_height()) return;"
      "float value=float((p.x*13+p.y*7+frame*19)%2048)/1024.0f-1.0f;out.write(half4(half(value)),p);}\n"
      "kernel void verify(texture2d<half,access::read> inp [[texture(0)]],texture2d<half,access::read> out [[texture(1)]],"
      "device float *errors [[buffer(0)]],uint2 p [[thread_position_in_grid]]) {"
      "if(p.x>=inp.get_width()||p.y>=inp.get_height()) return;"
      "half product=inp.read(p).x*half(0.5);half expected=max(product+half(0.25),half(0));"
      "errors[p.y*inp.get_width()+p.x]=abs(float(out.read(p).x)-float(expected));}\n";
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) { NSLog(@"probe shader: %@", error); return -11; }
    id<MTLComputePipelineState> fill = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"fill"] error:&error];
    id<MTLComputePipelineState> verify = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"verify"] error:&error];
    if (!fill || !verify) return -12;
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float width:spatial height:channels mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    NSMutableArray<id<MTLTexture>> *inputs = [NSMutableArray array], *outputs = [NSMutableArray array];
    NSMutableArray<id<MTLBuffer>> *checks = [NSMutableArray array];
    std::vector<IOSurfaceRef> surfaces;
    for (int i=0; i<slots; ++i) {
      IOSurfaceRef a=makeSurface(channels,spatial), b=makeSurface(channels,spatial);
      if (!a || !b) { if(a) CFRelease(a); if(b) CFRelease(b); for(auto s:surfaces) CFRelease(s); return -13; }
      surfaces.push_back(a); surfaces.push_back(b);
      int inRC=dlss_af_bind_surface((ane_e5rt_program_t *)handles[i],inputName,1,a);
      int outRC=dlss_af_bind_surface((ane_e5rt_program_t *)handles[i],outputName,0,b);
      fprintf(stderr,"surface slot %d bind input=%d output=%d ids=%u,%u\n",i,inRC,outRC,IOSurfaceGetID(a),IOSurfaceGetID(b));
      if(inRC || outRC) {for(auto s:surfaces) CFRelease(s); return -14;}
      id<MTLTexture> at=[device newTextureWithDescriptor:descriptor iosurface:a plane:0];
      id<MTLTexture> bt=[device newTextureWithDescriptor:descriptor iosurface:b plane:0];
      if(!at || !bt) {for(auto s:surfaces) CFRelease(s); return -15;}
      [inputs addObject:at]; [outputs addObject:bt];
      [checks addObject:[device newBufferWithLength:(size_t)channels*spatial*sizeof(float) options:MTLResourceStorageModeShared]];
    }
    dispatch_queue_t ane=dispatch_queue_create("dlss.aneforge.probe",DISPATCH_QUEUE_SERIAL);
    NSMutableArray *rows=[NSMutableArray array];
    __block int failed=0;
    auto began=std::chrono::steady_clock::now();
    // A completed slot refills itself, without waiting for the other slots.
    // A slot remains owned until the dependent GPU read and validation end.
    // The single ANE owner queue serializes handles and report mutation.
    dispatch_group_t group=dispatch_group_create();
    void (^submitSlot)(int,uint32_t);
    __block __weak void (^refillSlot)(int,uint32_t);
    submitSlot=^(int slot,uint32_t frame) {
        void *program=handles[slot];
        id<MTLTexture> input=inputs[slot],output=outputs[slot];
        id<MTLBuffer> check=checks[slot];
        id<MTLCommandBuffer> write=[gpu commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[write computeCommandEncoder];
        [encoder setComputePipelineState:fill];[encoder setTexture:input atIndex:0];
        [encoder setBytes:&frame length:sizeof(frame) atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(spatial,channels,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
        [write addCompletedHandler:^(id<MTLCommandBuffer> completedWrite) {
          if(completedWrite.status!=MTLCommandBufferStatusCompleted) {
            dispatch_async(ane,^{failed=-16;dispatch_group_leave(group);});return;
          }
          dispatch_async(ane,^{
            double aneStart=NSProcessInfo.processInfo.systemUptime;
            auto start=std::chrono::steady_clock::now();
            int rc=ane_e5rt_program_execute((ane_e5rt_program_t *)program);
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            double aneEnd=NSProcessInfo.processInfo.systemUptime;
            if(rc) {failed=rc;dispatch_group_leave(group);return;}
            id<MTLCommandBuffer> read=[gpu commandBuffer];
            id<MTLComputeCommandEncoder> e=[read computeCommandEncoder];
            [e setComputePipelineState:verify];[e setTexture:input atIndex:0];[e setTexture:output atIndex:1];[e setBuffer:check offset:0 atIndex:0];
            [e dispatchThreads:MTLSizeMake(spatial,channels,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];
            [read addCompletedHandler:^(id<MTLCommandBuffer> completedRead) {
              dispatch_async(ane,^{
                double maxError=0;int nonfinite=0;
                if(completedRead.status!=MTLCommandBufferStatusCompleted) {failed=-17;dispatch_group_leave(group);return;}
                const float *values=(const float *)check.contents;
                for(size_t j=0;j<(size_t)channels*spatial;++j) {
                  if(!std::isfinite(values[j])) ++nonfinite;
                  else maxError=std::max(maxError,(double)values[j]);
                }
                [rows addObject:@{@"frame":@(frame),@"slot":@(slot),@"aneWallMS":@(ms),
                  @"gpuWriteMS":@((completedWrite.GPUEndTime-completedWrite.GPUStartTime)*1000),
                  @"gpuWriteStart":@(completedWrite.GPUStartTime),@"gpuWriteEnd":@(completedWrite.GPUEndTime),
                  @"aneStart":@(aneStart),@"aneEnd":@(aneEnd),
                  @"gpuReadStart":@(completedRead.GPUStartTime),@"gpuReadEnd":@(completedRead.GPUEndTime),
                  @"gpuVerifyMS":@((completedRead.GPUEndTime-completedRead.GPUStartTime)*1000),
                  @"maxAbsError":@(maxError),@"nonfinite":@(nonfinite)}];
                if(nonfinite || maxError>0.001) failed=-18;
                if(!failed && frame+(uint32_t)slots<(uint32_t)iterations)
                  refillSlot(slot,frame+(uint32_t)slots);
                else dispatch_group_leave(group);
              });
            }];
            [read commit];
          });
        }];
        [write commit];
    };
    refillSlot=submitSlot;
    for(int slot=0;slot<slots;++slot) {
      dispatch_group_enter(group);
      dispatch_async(ane,^{submitSlot(slot,(uint32_t)slot);});
    }
    // Only the offline probe caller waits. No waits occur in Metal callbacks.
    dispatch_group_wait(group,DISPATCH_TIME_FOREVER);
    submitSlot=nil;
    double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
    NSDictionary *report=@{@"scope":@"ANE-only affine IOSurface interoperability, not full NR inference",
      @"scheduling":@"continuous-slot-refill",@"device":device.name,@"slots":@(slots),@"iterations":@(rows.count),@"wallMS":@(elapsed),
      @"shape":@[@1,@(channels),@1,@(spatial)],@"deviceMask":@4,@"externalTensorCopies":@0,
      @"surfaceIDAndAddressIdentity":@YES,@"result":@(failed),@"frames":rows};
    NSData *json=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    [json writeToFile:[NSString stringWithUTF8String:reportPath] atomically:YES];
    for(auto s:surfaces) CFRelease(s);
    return failed;
  }
}

// Probe utilities: returned surface has +1 ownership. Only the validation driver
// reads its final CPU view; inference transfers use the Metal/ANE bindings above.
extern "C" IOSurfaceRef dlss_af_make_surface(int channels,int spatial) {
  return channels>0 && spatial>0 ? makeSurface(channels,spatial) : nullptr;
}
extern "C" void dlss_af_release_surface(IOSurfaceRef surface) { if(surface) CFRelease(surface); }
extern "C" int dlss_af_fill_surface(IOSurfaceRef surface,uint32_t frame) {
  @autoreleasepool {
    if(!surface) return -1;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    auto *descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float width:IOSurfaceGetWidth(surface) height:IOSurfaceGetHeight(surface) mipmapped:NO];
    descriptor.storageMode=MTLStorageModeShared;descriptor.usage=MTLTextureUsageShaderWrite|MTLTextureUsageShaderRead;
    id<MTLTexture> texture=[device newTextureWithDescriptor:descriptor iosurface:surface plane:0];
    NSString *code=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void fill(texture2d<half,access::write> o [[texture(0)]],constant uint &frame [[buffer(0)]],uint2 p [[thread_position_in_grid]]) {if(p.x>=o.get_width()||p.y>=o.get_height()) return;float v=float((p.x*13+p.y*7+frame*19)%2048)/32768.0f-0.03125f;o.write(half4(half(v)),p);}";
    NSError *error=nil;
    id<MTLLibrary> library=[device newLibraryWithSource:code options:nil error:&error];
    id<MTLComputePipelineState> state=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"fill"] error:&error];
    if(!texture||!state) return -2;
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:state];[encoder setTexture:texture atIndex:0];[encoder setBytes:&frame length:sizeof(frame) atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(texture.width,texture.height,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
    [command commit];[command waitUntilCompleted];
    return command.status==MTLCommandBufferStatusCompleted ? 0 : -3;
  }
}

// Native hot-path API. Python/ANEForge is only needed to lower the graph ahead
// of time; these calls load and execute its MIL directory directly in-process.
extern "C" void *dlss_ane_compile_directory(const char *directory,const char *inputName,
                                             const char *outputName,size_t inputBytes,
                                             size_t outputBytes) {
  return dlss_ane_compile_directory_inputs(directory,&inputName,&inputBytes,1,outputName,outputBytes);
}

extern "C" void *dlss_ane_compile_directory_inputs(const char *directory,
  const char *const *inputNames,const size_t *inputBytes,size_t inputCount,
  const char *outputName,size_t outputBytes) {
  return dlss_ane_compile_directory_ports(directory,inputNames,inputBytes,inputCount,&outputName,&outputBytes,1);
}
extern "C" void *dlss_ane_compile_directory_ports(const char *directory,
  const char *const *inputNames,const size_t *inputBytes,size_t inputCount,
  const char *const *outputNames,const size_t *outputBytes,size_t outputCount) {
  @autoreleasepool {
    if(!directory||!inputNames||!inputBytes||!inputCount||!outputNames||!outputBytes||!outputCount)return nullptr;
    for(size_t i=0;i<outputCount;++i)if(!outputNames[i]||!outputBytes[i])return nullptr;
    for(size_t i=0;i<inputCount;++i)if(!inputNames[i]||!inputBytes[i])return nullptr;
    NSString *root=[NSString stringWithUTF8String:directory];
    NSString *mil=[root stringByAppendingPathComponent:@"model.mil"];
    NSString *cache=[root stringByAppendingPathComponent:@"cache"];
    if(![[NSFileManager defaultManager] fileExistsAtPath:mil])return nullptr;
    return ane_e5rt_program_compile(mil.UTF8String,cache.UTF8String,4,
      inputNames,inputBytes,inputCount,outputNames,outputBytes,outputCount);
  }
}
extern "C" int dlss_ane_execute(void *handle) {
  auto *program=(ane_e5rt_program_t *)handle;
  return program && program->device_mask==4 ? ane_e5rt_program_execute(program) : -1;
}
extern "C" void dlss_ane_destroy(void *handle) {
  if(handle) ane_e5rt_program_release((ane_e5rt_program_t *)handle);
}
