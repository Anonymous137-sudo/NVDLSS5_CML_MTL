// Complete captured-stage experiment. This synchronous entry point belongs on
// an inference worker, never the capture callback, presenter, or main thread.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#include "ANEForgeBridge.h"

@interface DLSSHybridFFN : NSObject {
@public
  IOSurfaceRef surfaces[5];
  void *expandProgram, *branchProgram;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> gate, pack, unpack;
  id<MTLTexture> expanded, published, input, output;
}
@end
@implementation DLSSHybridFFN
- (void)dealloc { for(int i=0;i<5;++i) if(surfaces[i]) CFRelease(surfaces[i]); }
@end

static void *createHybrid(void *expand,void *branch,
  const char *expandInput,const char *expandOutput,const char *branchInput,
  const char *branchSkip,const char *branchOutput,int tokens,const char *qkvOutput) {
  @autoreleasepool {
    if(!expand||!branch||tokens<32||tokens%32) return nullptr;
    DLSSHybridFFN *context=[DLSSHybridFFN new];
    context->expandProgram=expand;context->branchProgram=branch;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    context->queue=[device newCommandQueue];
    for(int i=0;i<(qkvOutput?5:4);++i) {
      int channels=i==4?3072:((i==1||i==2)?4096:1024);
      NSDictionary *properties=@{(id)kIOSurfaceWidth:@(tokens),(id)kIOSurfaceHeight:@(channels),
        (id)kIOSurfaceBytesPerElement:@2,(id)kIOSurfaceBytesPerRow:@(tokens*2),
        (id)kIOSurfaceAllocSize:@((size_t)channels*tokens*2)};
      context->surfaces[i]=IOSurfaceCreate((__bridge CFDictionaryRef)properties);
      if(!context->surfaces[i]) return nullptr;
    }
    if(dlss_af_bind_surface(expand,expandInput,1,context->surfaces[0]) ||
       dlss_af_bind_surface(expand,expandOutput,0,context->surfaces[1]) ||
       dlss_af_bind_surface(branch,branchInput,1,context->surfaces[2]) ||
       dlss_af_bind_surface(branch,branchSkip,1,context->surfaces[0]) ||
       dlss_af_bind_surface(branch,branchOutput,0,context->surfaces[3])) return nullptr;
    if(qkvOutput&&dlss_af_bind_surface(branch,qkvOutput,0,context->surfaces[4]))return nullptr;
    MTLTextureDescriptor *desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
      width:tokens height:4096 mipmapped:NO];
    desc.storageMode=MTLStorageModeShared;desc.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
    context->expanded=[device newTextureWithDescriptor:desc iosurface:context->surfaces[1] plane:0];
    context->published=[device newTextureWithDescriptor:desc iosurface:context->surfaces[2] plane:0];
    desc.height=1024;
    context->input=[device newTextureWithDescriptor:desc iosurface:context->surfaces[0] plane:0];
    context->output=[device newTextureWithDescriptor:desc iosurface:context->surfaces[3] plane:0];
    NSString *source=@"#include <metal_stdlib>\nusing namespace metal;\n"
      "half publish(half input) {uint b=as_type<uint>(float(input));uint sign=b&0x80000000u;b^=sign;uchar q;"
      "if(b>=(543u<<21))q=0x7e;else if(b<(121u<<23)){b=as_type<uint>(as_type<float>(b)+as_type<float>(141u<<23));q=uchar(b-(141u<<23));}"
      "else{uint odd=(b>>20)&1u;b+=((7u-127u)<<23)+0x7ffffu;b+=odd;q=uchar(b>>20);}"
      "half v=as_type<half>(ushort((q&127)<<7))*half(256);return sign ? -v:v;}\n"
      "kernel void gate(texture2d<half,access::read> input [[texture(0)]],texture2d<half,access::write> output [[texture(1)]],"
      "uint2 p [[thread_position_in_grid]]){if(p.x>=input.get_width()||p.y>=input.get_height())return;"
      "half x=input.read(p).x;float c=clamp(float(x),-4.0f,4.0f);"
      "half l=half(fma(abs(c),-0.055908203125f,0.447265625f));"
      "half g=half(fma(c,float(l),0.89453125f));half a=half(float(x)*float(g));output.write(half4(publish(a)),p);}\n"
      "kernel void pack(device const half* input [[buffer(0)]],texture2d<half,access::write> output [[texture(0)]],"
      "uint2 p [[thread_position_in_grid]]){if(p.x>=output.get_width()||p.y>=output.get_height())return;"
      "output.write(half4(input[p.x*output.get_height()+p.y]),p);}\n"
      "kernel void unpack(texture2d<half,access::read> input [[texture(0)]],device half* output [[buffer(0)]],"
      "uint2 p [[thread_position_in_grid]]){if(p.x>=input.get_width()||p.y>=input.get_height())return;"
      "output[p.x*input.get_height()+p.y]=input.read(p).x;}";
    NSError *error=nil;
    MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if(!library){NSLog(@"Hybrid FFN shader: %@",error);return nullptr;}
    context->gate=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gate"] error:&error];
    context->pack=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"pack"] error:&error];
    context->unpack=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"unpack"] error:&error];
    if(!context->input||!context->output||!context->pack||!context->unpack)return nullptr;
    if(!context->expanded||!context->published||!context->gate||!context->queue)return nullptr;
    return (__bridge_retained void *)context;
  }
}
extern "C" void *dlss_af_hybrid_create(void *expand,void *branch,
  const char *expandInput,const char *expandOutput,const char *branchInput,
  const char *branchSkip,const char *branchOutput,int tokens) {
  return createHybrid(expand,branch,expandInput,expandOutput,branchInput,branchSkip,branchOutput,tokens,nullptr);
}
extern "C" void *dlss_af_hybrid_create_qkv(void *expand,void *branch,
  const char *expandInput,const char *expandOutput,const char *branchInput,
  const char *branchSkip,const char *branchOutput,const char *qkvOutput,int tokens) {
  if(!qkvOutput)return nullptr;
  return createHybrid(expand,branch,expandInput,expandOutput,branchInput,branchSkip,branchOutput,tokens,qkvOutput);
}
extern "C" int dlss_af_hybrid_execute(void *handle,double *timings) {
  @autoreleasepool {
    if(!handle||!timings)return -1;
    DLSSHybridFFN *context=(__bridge DLSSHybridFFN *)handle;
    double start=NSProcessInfo.processInfo.systemUptime;
    int rc=dlss_ane_execute(context->expandProgram);if(rc)return rc;
    double expandedAt=NSProcessInfo.processInfo.systemUptime;
    id<MTLCommandBuffer> command=[context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if(!command||!encoder)return -2;
    [encoder setComputePipelineState:context->gate];[encoder setTexture:context->expanded atIndex:0];
    [encoder setTexture:context->published atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(context->expanded.width,context->expanded.height,1)
      threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
    [command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted)return -3;
    double gateAt=NSProcessInfo.processInfo.systemUptime;
    rc=dlss_ane_execute(context->branchProgram);if(rc)return rc;
    double end=NSProcessInfo.processInfo.systemUptime;
    timings[0]=(expandedAt-start)*1000;timings[1]=(gateAt-expandedAt)*1000;
    timings[2]=(end-gateAt)*1000;timings[3]=(end-start)*1000;
    timings[4]=(command.GPUEndTime-command.GPUStartTime)*1000;
    return 0;
  }
}
// GPU-side NHWC<->NCHW conversion. The caller retains and synchronizes the
// contiguous source MLX array before calling, and keeps output alive until its
// MLX consumer completes. This function serializes one context on its worker.
static id<MTLCommandBuffer> layoutCommand(DLSSHybridFFN *context,id<MTLBuffer> buffer,bool pack) {
  id<MTLTexture> texture=pack?context->input:context->output;
  id<MTLCommandBuffer> command=[context->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  if(!command||!encoder)return nil;
  [encoder setComputePipelineState:pack?context->pack:context->unpack];
  [encoder setBuffer:buffer offset:0 atIndex:0];[encoder setTexture:texture atIndex:0];
  [encoder dispatchThreads:MTLSizeMake(texture.width,texture.height,1)
    threadsPerThreadgroup:MTLSizeMake(8,32,1)];[encoder endEncoding];
  return command;
}
extern "C" int dlss_af_hybrid_execute_buffers(void *handle,void *source,void *destination,double *timings) {
  @autoreleasepool {
    if(!handle||!source||!timings)return -1;
    DLSSHybridFFN *context=(__bridge DLSSHybridFFN *)handle;
    id<MTLBuffer> input=(__bridge id<MTLBuffer>)source,output=(__bridge id<MTLBuffer>)destination;
    size_t bytes=context->input.width*1024*2;
    if(input.length<bytes||input.device.registryID!=context->queue.device.registryID||
       (output&&(input==output||output.length<bytes||output.device.registryID!=context->queue.device.registryID)))return -2;
    double start=NSProcessInfo.processInfo.systemUptime;
    id<MTLCommandBuffer> pack=layoutCommand(context,input,true);
    if(!pack)return -3;
    [pack commit];[pack waitUntilCompleted];
    if(pack.status!=MTLCommandBufferStatusCompleted)return -4;
    double packedAt=NSProcessInfo.processInfo.systemUptime,inner[5];
    int rc=dlss_af_hybrid_execute(handle,inner);if(rc)return rc;
    double computedAt=NSProcessInfo.processInfo.systemUptime;
    id<MTLCommandBuffer> unpack=output?layoutCommand(context,output,false):nil;
    if(output) {
      if(!unpack)return -5;
      [unpack commit];[unpack waitUntilCompleted];
      if(unpack.status!=MTLCommandBufferStatusCompleted)return -6;
    }
    double end=NSProcessInfo.processInfo.systemUptime;
    timings[0]=(end-start)*1000;timings[1]=(packedAt-start)*1000;
    timings[2]=(computedAt-packedAt)*1000;timings[3]=(end-computedAt)*1000;
    timings[4]=(pack.GPUEndTime-pack.GPUStartTime)*1000;timings[5]=inner[4];
    timings[6]=(unpack.GPUEndTime-unpack.GPUStartTime)*1000;
    timings[7]=inner[0];timings[8]=inner[2];
    return 0;
  }
}
extern "C" IOSurfaceRef dlss_af_hybrid_copy_output_surface(void *handle) {
  if(!handle)return nullptr;
  DLSSHybridFFN *context=(__bridge DLSSHybridFFN *)handle;
  return (IOSurfaceRef)CFRetain(context->surfaces[3]);
}
extern "C" IOSurfaceRef dlss_af_hybrid_copy_qkv_surface(void *handle) {
  if(!handle)return nullptr;
  DLSSHybridFFN *context=(__bridge DLSSHybridFFN *)handle;
  return context->surfaces[4]?(IOSurfaceRef)CFRetain(context->surfaces[4]):nullptr;
}
extern "C" void dlss_af_hybrid_destroy(void *handle) {
  if(handle) {DLSSHybridFFN *context=CFBridgingRelease(handle);(void)context;}
}
