#import <AppKit/AppKit.h>
#include <obs.h>
#include <stdio.h>
#include <dlfcn.h>
#include <CoreVideo/CoreVideo.h>

int main(int argc,char **argv) {
    @autoreleasepool {
        if(argc<2) return 2;
        [NSApplication sharedApplication];
        if(!obs_startup("en-US",NULL,NULL)) return 3;
        obs_add_data_path(NEURAL_POST_OBS_APP "/Contents/Frameworks/libobs.framework/Resources");
        struct obs_video_info video={0};
        video.graphics_module=NEURAL_POST_OBS_APP "/Contents/Frameworks/libobs-metal.dylib";
        video.fps_num=60; video.fps_den=1; video.base_width=1280; video.base_height=720;
        video.output_width=1280; video.output_height=720; video.output_format=VIDEO_FORMAT_NV12;
        video.colorspace=VIDEO_CS_709; video.range=VIDEO_RANGE_PARTIAL; video.gpu_conversion=true;
        video.scale_type=OBS_SCALE_BILINEAR;
        if(obs_reset_video(&video)!=OBS_VIDEO_SUCCESS) return 12;
        obs_module_t *module=NULL;
        int result=obs_open_module(&module,argv[1],".");
        if(result!=MODULE_SUCCESS || !obs_init_module(module)) {
            fprintf(stderr,"module open/init failed: %d\n",result); obs_shutdown(); return 4;
        }
        for(int i=0;i<5;i++) {
            obs_data_t *settings=obs_data_create();
            // Empty device: validate lifecycle without starting camera/TCC.
            obs_source_t *source=obs_source_create("mac_neural_hdmi","Neural post load test",settings,NULL);
            if(!source) return 5;
            if(obs_source_get_width(source)!=3840||obs_source_get_height(source)!=2160) return 6;
            obs_source_release(source); obs_data_release(settings);
        }
        if(argc>2) {
            NSString *library=[[[@(argv[1]) stringByDeletingLastPathComponent] stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"Frameworks/libNeuralPost.dylib"];
            void *dylib=dlopen(library.fileSystemRepresentation,RTLD_NOW);
            if(!dylib) { fprintf(stderr,"%s\n",dlerror()); return 7; }
            void *(*create)(const char *,const char *)=dlsym(dylib,"np_create");
            void (*submit)(void *,void *,uint64_t,double)=dlsym(dylib,"np_submit");
            void *(*acquire)(void *)=dlsym(dylib,"np_frame_acquire");
            void *(*pixels)(void *)=dlsym(dylib,"np_frame_pixels");
            void (*releaseFrame)(void *)=dlsym(dylib,"np_frame_release");
            void (*destroy)(void *)=dlsym(dylib,"np_destroy");
            void (*status)(void *,char *,int32_t)=dlsym(dylib,"np_status");
            if(!create||!submit||!acquire||!pixels||!releaseFrame||!destroy||!status) return 8;
            void *(*hybrid)(const char *,const char *)=dlsym(dylib,"np_create_hybrid");
            if(argc>3 && !hybrid) return 13;
            void *engine=argc>3 ? hybrid(argv[2],argv[3]) : create(argv[2],"");
            CVPixelBufferRef input=NULL;
            NSDictionary *attrs=@{(id)kCVPixelBufferIOSurfacePropertiesKey:@{},(id)kCVPixelBufferMetalCompatibilityKey:@YES};
            if(CVPixelBufferCreate(NULL,1280,720,kCVPixelFormatType_32BGRA,(__bridge CFDictionaryRef)attrs,&input)!=kCVReturnSuccess) return 9;
            CVPixelBufferLockBaseAddress(input,0);
            memset(CVPixelBufferGetBaseAddress(input),127,CVPixelBufferGetDataSize(input));
            CVPixelBufferUnlockBaseAddress(input,0);
            submit(engine,input,1,NSProcessInfo.processInfo.systemUptime);
            void *frame=NULL;
            double until=NSProcessInfo.processInfo.systemUptime+30;
            while(!(frame=acquire(engine)) && NSProcessInfo.processInfo.systemUptime<until)
                [NSRunLoop.currentRunLoop runUntilDate:[NSDate dateWithTimeIntervalSinceNow:.01]];
            char text[1024]; status(engine,text,sizeof(text)); puts(text);
            if(!frame) return 10;
            if(argc>3 && !strstr(text,"appearance=1")) return 14;
            CVPixelBufferRef output=pixels(frame);
            if(CVPixelBufferGetWidth(output)!=3840||CVPixelBufferGetHeight(output)!=2160) return 11;
            releaseFrame(frame); destroy(engine); CFRelease(input);
            puts("PASS: packaged MLX weights executed and produced a completed 4K IOSurface");
        }
        obs_shutdown();
        puts("PASS: OBS module loaded and five source lifecycles completed without capture");
    }
    return 0;
}
