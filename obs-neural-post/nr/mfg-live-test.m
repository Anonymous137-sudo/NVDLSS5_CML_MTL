#import <AppKit/AppKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <obs.h>

@interface Clip : NSObject
@property AVAssetWriter *writer;
@property AVAssetWriterInput *input;
@property AVAssetWriterInputPixelBufferAdaptor *adaptor;
@property dispatch_queue_t queue;
@property BOOL busy;
@property uint64_t start,frames,dropped;
@property NSMutableSet *signatures;
@end
@implementation Clip
@end
static void record_frame(void *context,struct video_data *video) {
 Clip *clip=(__bridge Clip *)context;
 @synchronized(clip){if(clip.busy){clip.dropped++;return;}clip.busy=YES;}
 CVPixelBufferRef pixels=NULL;
 NSDictionary *attrs=@{(id)kCVPixelBufferIOSurfacePropertiesKey:@{},(id)kCVPixelBufferMetalCompatibilityKey:@YES};
 if(CVPixelBufferCreate(NULL,1920,1080,kCVPixelFormatType_32BGRA,(__bridge CFDictionaryRef)attrs,&pixels)!=kCVReturnSuccess){@synchronized(clip){clip.busy=NO;}return;}
 CVPixelBufferLockBaseAddress(pixels,0);
 uint8_t *dst=CVPixelBufferGetBaseAddress(pixels);size_t row=CVPixelBufferGetBytesPerRow(pixels);
 for(int y=0;y<1080;y++)memcpy(dst+y*row,video->data[0]+y*video->linesize[0],1920*4);
 uint64_t hash=1469598103934665603ULL;
 for(int y=13;y<1080;y+=29)for(int x=7;x<1920;x+=31){hash^=dst[y*row+x*4];hash*=1099511628211ULL;}
 CVPixelBufferUnlockBaseAddress(pixels,0);
 uint64_t pts=video->timestamp;
 dispatch_async(clip.queue, ^{
  @autoreleasepool {
   if(!clip.start)clip.start=pts;
   if(clip.input.readyForMoreMediaData && [clip.adaptor appendPixelBuffer:pixels withPresentationTime:CMTimeMake((int64_t)(pts-clip.start),1000000000)]) {clip.frames++;[clip.signatures addObject:@(hash)];}else clip.dropped++;
   CFRelease(pixels);@synchronized(clip){clip.busy=NO;}
  }
 });
}
static void pump(double seconds) {
 double end=NSProcessInfo.processInfo.systemUptime+seconds;
 while(NSProcessInfo.processInfo.systemUptime<end)[NSRunLoop.currentRunLoop runUntilDate:[NSDate dateWithTimeIntervalSinceNow:.02]];
}
static Clip *start_clip(NSString *path) {
 Clip *clip=[Clip new];clip.queue=dispatch_queue_create("mfg.test.encode",DISPATCH_QUEUE_SERIAL);clip.signatures=[NSMutableSet set];
 NSError *error=nil;clip.writer=[[AVAssetWriter alloc] initWithURL:[NSURL fileURLWithPath:path] fileType:AVFileTypeQuickTimeMovie error:&error];
 clip.input=[AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:@{AVVideoCodecKey:AVVideoCodecTypeHEVC,AVVideoWidthKey:@1920,AVVideoHeightKey:@1080}];clip.input.expectsMediaDataInRealTime=YES;
 clip.adaptor=[AVAssetWriterInputPixelBufferAdaptor assetWriterInputPixelBufferAdaptorWithAssetWriterInput:clip.input sourcePixelBufferAttributes:@{(id)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_32BGRA),(id)kCVPixelBufferWidthKey:@1920,(id)kCVPixelBufferHeightKey:@1080}];
 [clip.writer addInput:clip.input];if(![clip.writer startWriting]){NSLog(@"encode init %@",clip.writer.error);return nil;}[clip.writer startSessionAtSourceTime:kCMTimeZero];
 struct video_scale_info format={0};format.format=VIDEO_FORMAT_BGRA;format.width=1920;format.height=1080;format.colorspace=VIDEO_CS_709;format.range=VIDEO_RANGE_FULL;
 obs_add_raw_video_callback(&format,record_frame,(__bridge void *)clip);return clip;
}
static NSDictionary *stop_clip(Clip *clip) {
 obs_remove_raw_video_callback(record_frame,(__bridge void *)clip);dispatch_sync(clip.queue,^{});
 [clip.input markAsFinished];dispatch_semaphore_t done=dispatch_semaphore_create(0);[clip.writer finishWritingWithCompletionHandler:^{dispatch_semaphore_signal(done);}];
 dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,20*NSEC_PER_SEC));
 return @{ @"frames":@(clip.frames),@"droppedRecordingFrames":@(clip.dropped),@"distinctSampledPixelSignatures":@(clip.signatures.count),@"status":@(clip.writer.status),@"error":clip.writer.error.localizedDescription ?: @""};
}
int main(int argc,char **argv) {@autoreleasepool {
 if(argc!=4)return 2;[NSApplication sharedApplication];if(!obs_startup("en-US",NULL,NULL))return 3;
 NSString *app=@(argv[1]),*plugin=@(argv[2]),*output=@(argv[3]);
 [[NSFileManager defaultManager] createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil];
 obs_add_data_path([[app stringByAppendingPathComponent:@"Contents/Frameworks/libobs.framework/Resources"] UTF8String]);
 NSString *graphics=[app stringByAppendingPathComponent:@"Contents/Frameworks/libobs-metal.dylib"];
 struct obs_video_info video={0};video.graphics_module=graphics.UTF8String;video.fps_num=60;video.fps_den=1;video.base_width=1920;video.base_height=1080;video.output_width=1920;video.output_height=1080;video.output_format=VIDEO_FORMAT_BGRA;video.colorspace=VIDEO_CS_709;video.range=VIDEO_RANGE_FULL;video.gpu_conversion=false;video.scale_type=OBS_SCALE_BILINEAR;
 if(obs_reset_video(&video)!=OBS_VIDEO_SUCCESS)return 4;
 obs_module_t *module=NULL;if(obs_open_module(&module,plugin.UTF8String,".")!=MODULE_SUCCESS || !obs_init_module(module))return 5;
 obs_data_t *settings=obs_data_create();obs_data_set_string(settings,"mode","31-480");
 if(getenv("DLSS_CAPTURE_DEVICE"))obs_data_set_string(settings,"device",getenv("DLSS_CAPTURE_DEVICE"));
 if(getenv("DLSS_MFG_BACKEND"))obs_data_set_string(settings,"frameGenerationBackend",getenv("DLSS_MFG_BACKEND"));
 if(getenv("DLSS_CAPTURE_GUIDED"))obs_data_set_bool(settings,"captureGuided",true);
 obs_source_t *source=obs_source_create("mac_dlss5_nr","MFG live test",settings,NULL);if(!source)return 6;
 obs_set_output_source(0,source);
 NSArray *cases=@[@[@"31-480",@1],@[@"31-480",@2],@[@"31-480",@3],@[@"31-480",@4],@[@"31then71-480",@4],@[@"71x2-480",@4]];
 const char *subset=getenv("DLSS_MFG_CASES");
 if(subset && !strcmp(subset,"4x"))cases=@[@[@"31-480",@4],@[@"31then71-480",@4],@[@"71x2-480",@4]];
 if(getenv("DLSS_MFG_NATIVE_TEST"))cases=@[@[@"31-480",@4]];
 if(getenv("DLSS_MFG_GAME_TEST"))cases=@[@[@"71x2-480",@4]];
 if(getenv("DLSS_MFG_SMOOTH_BASELINE"))cases=@[@[@"31then71-480",@4],@[@"71x2-480",@4]];
 if(getenv("DLSS_MFG6_TEST"))cases=@[@[@"31then71-480",@4,@360],@[@"31then71-480",@5,@360],@[@"31then71-480",@6,@360],@[@"71x2-480",@6,@360],@[@"31-480",@6,@360],@[@"31then71-480",@6,@540],@[@"71x2-480",@6,@540],@[@"31then71-480",@6,@720]];
 if(getenv("DLSS_MFG_CASES_JSON")){NSData *data=[@(getenv("DLSS_MFG_CASES_JSON")) dataUsingEncoding:NSUTF8StringEncoding];NSArray *requested=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];if(![requested isKindOfClass:NSArray.class] || !requested.count)return 12;cases=requested;}
 NSString *statusFile=[output stringByAppendingPathComponent:@"runtime.json"];
 NSMutableArray *reports=[NSMutableArray array];
 for(NSArray *entry in cases) {
  NSString *mode=entry[0];int factor=[entry[1] intValue];int height=entry.count>2 ? [entry[2] intValue] : 360;NSString *name=getenv("DLSS_MFG6_TEST") ? [NSString stringWithFormat:@"%@-%dx-%dp",mode,factor,height] : [NSString stringWithFormat:@"%@-%dx",mode,factor];
  if(entry.count>3)obs_data_set_string(settings,"frameGenerationBackend",[entry[3] UTF8String]);
  if(entry.count>4)obs_data_set_bool(settings,"captureGuided",[entry[4] boolValue]);
  name=[name stringByAppendingFormat:@"-%s-%@",obs_data_get_string(settings,"frameGenerationBackend"),obs_data_get_bool(settings,"captureGuided") ? @"guided":@"synthetic"];
  obs_data_set_string(settings,"mode",mode.UTF8String);obs_data_set_int(settings,"frameGenerationFactor",factor);obs_data_set_int(settings,"frameGenerationHeight",height);obs_source_update(source,settings);
  pump(4);
  if(!strcmp(obs_data_get_string(settings,"frameGenerationBackend"),"apple")) {
   double deadline=NSProcessInfo.processInfo.systemUptime+30;
   while(NSProcessInfo.processInfo.systemUptime<deadline) {
    NSDictionary *warm=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:statusFile] ?: [NSData data] options:0 error:nil];
    if([warm[@"nativeFGState"] isEqual:@"ready"] && [warm[@"generatedFrames"] unsignedLongLongValue]>0)break;
    pump(0.25);
   }
  }
  obs_properties_t *diagnostics=obs_source_properties(source);NSLog(@"Runtime before recording: %s",obs_property_description(obs_properties_get(diagnostics,"status")));obs_properties_destroy(diagnostics);
  if(getenv("DLSS_MFG_NATIVE_TEST") || getenv("DLSS_MFG_GAME_TEST") || getenv("DLSS_MFG_SMOOTH_BASELINE") || getenv("DLSS_MFG6_TEST")){obs_properties_t *props=obs_source_properties(source);obs_property_button_clicked(obs_properties_get(props,"preview"),source);obs_properties_destroy(props);pump(2);}
  NSDictionary *before=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:statusFile] ?: [NSData data] options:0 error:nil] ?: @{};
  Clip *clip=start_clip([output stringByAppendingPathComponent:[name stringByAppendingString:@".mov"]]);if(!clip)return 7;
  pump(10);NSDictionary *recording=stop_clip(clip);pump(10);
  NSMutableDictionary *report=[[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:statusFile] ?: [NSData data] options:NSJSONReadingMutableContainers error:nil] mutableCopy];
  if(!report || [report[@"mfgFactor"] intValue]!=factor || [report[@"failed"] intValue] || [report[@"mfgFailures"] intValue] || [report[@"captureStarts"] intValue]!=1 || [report[@"playbackQueueDepth"] intValue]>12) {NSLog(@"FAIL runtime %@ %@",name,report);return 8;}
  if(factor>1 && [report[@"obsSelectedGenerated"] longLongValue]<=[before[@"obsSelectedGenerated"] longLongValue]){NSLog(@"FAIL no generated selections %@",report);return 9;}
  if(getenv("DLSS_MFG_VALIDATE") && ![report[@"motionPolicy"] isEqual:@"guarded-single-trajectory"] && [report[@"warpParityMaxByteDelta"] intValue]<0)return 11;
  report[@"recording"]=recording;report[@"case"]=name;report[@"sampleStart"]=before;
  [reports addObject:report];[[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil] writeToFile:[output stringByAppendingPathComponent:[name stringByAppendingString:@".json"]] atomically:YES];
  NSLog(@"PASS %@: %@ base FPS, %@ generated total, %@ queue depth",name,report[@"processedFPS"],report[@"generatedFrames"],report[@"playbackQueueDepth"]);
 }
 obs_data_set_int(settings,"frameGenerationFactor",1);obs_source_update(source,settings);pump(6);
 NSDictionary *off=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:statusFile] options:0 error:nil];
 if([off[@"mfgFactor"] intValue]!=1 || [off[@"playbackQueueDepth"] intValue]!=0)return 10;
 [[NSJSONSerialization dataWithJSONObject:reports options:NSJSONWritingPrettyPrinted error:nil] writeToFile:[output stringByAppendingPathComponent:@"results.json"] atomically:YES];
 obs_set_output_source(0,NULL);obs_source_release(source);obs_data_release(settings);obs_shutdown();NSLog(@"PASS requested MFG live cases and reset to off");return 0;
}}
