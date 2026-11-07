#import <AppKit/AppKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <obs.h>
#include <dlfcn.h>
static void pump(double seconds) {
 double end=NSProcessInfo.processInfo.systemUptime+seconds;
 while(NSProcessInfo.processInfo.systemUptime<end)[NSRunLoop.currentRunLoop runUntilDate:[NSDate dateWithTimeIntervalSinceNow:.02]];
}
int main(int argc,char **argv) {@autoreleasepool {
 if(argc!=4)return 2;
 [NSApplication sharedApplication];
 if(!obs_startup("en-US",NULL,NULL))return 3;
 NSString *app=@(argv[1]),*plugin=@(argv[2]),*output=@(argv[3]);
 [[NSFileManager defaultManager] createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:nil];
 obs_add_data_path([[app stringByAppendingPathComponent:@"Contents/Frameworks/libobs.framework/Resources"] UTF8String]);
 NSString *graphics=[app stringByAppendingPathComponent:@"Contents/Frameworks/libobs-metal.dylib"];
 struct obs_video_info video={0};video.graphics_module=graphics.UTF8String;video.fps_num=60;video.fps_den=1;video.base_width=1920;video.base_height=1080;video.output_width=1920;video.output_height=1080;video.output_format=VIDEO_FORMAT_NV12;video.colorspace=VIDEO_CS_709;video.range=VIDEO_RANGE_PARTIAL;video.gpu_conversion=true;video.scale_type=OBS_SCALE_BILINEAR;
 if(obs_reset_video(&video)!=OBS_VIDEO_SUCCESS)return 4;
 obs_module_t *module=NULL;
 if(obs_open_module(&module,plugin.UTF8String,".")!=MODULE_SUCCESS || !obs_init_module(module))return 5;
 obs_data_t *settings=obs_data_create();
 obs_source_t *source=obs_source_create("mac_dlss5_nr","DLSS live test",settings,NULL);
 if(!source)return 6;
 if(obs_source_get_width(source)!=1920 || obs_source_get_height(source)!=1080)return 7;
 obs_properties_t *props=obs_source_properties(source);
 if(!obs_properties_get(props,"mode") || !obs_properties_get(props,"style") || !obs_properties_get(props,"skinStructure"))return 8;
 obs_property_t *mode=obs_properties_get(props,"mode");if(obs_property_list_item_count(mode)!=3)return 9;
 obs_properties_destroy(props);
 obs_set_output_source(0,source);
 NSArray *modes=getenv("DLSS_NR_UI_TEST_ONLY") ? @[@"31-480"] : @[@"31then71-480",@"71x2-480",@"31-480"];
 NSMutableArray *reports=[NSMutableArray array];
 for(NSString *selected in modes) {
  obs_data_set_string(settings,"mode",selected.UTF8String);obs_source_update(source,settings);
  pump(18);
  NSString *file=[output stringByAppendingPathComponent:@"runtime.json"];
  NSDictionary *report=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:file] ?: [NSData data] options:0 error:nil];
  if(![report[@"modeID"] isEqual:selected] || [report[@"completed"] intValue]<30 || [report[@"failed"] intValue]!=0 || [report[@"outputWidth"] intValue]!=1920 || [report[@"outputHeight"] intValue]!=1080){NSLog(@"FAILED %@ %@",selected,report);return 10;}
  if([report[@"captureStarts"] intValue]!=1 || [report[@"captureDrops"] intValue]!=0)return 13;
  NSDictionary *stages=report[@"stages"];
  if(![stages[@"pass1MS"] doubleValue] || ([selected isEqual:@"31-480"] ? stages[@"pass2MS"]!=nil : stages[@"pass2MS"]==nil))return 11;
  [reports addObject:report];
  [[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil] writeToFile:[output stringByAppendingPathComponent:[selected stringByAppendingString:@".json"]] atomically:YES];
  NSLog(@"PASS live %@: %@ms, %@ processed FPS",selected,report[@"processingMedianMS"],report[@"processedFPS"]);
 }
 // Exercise actual parameter changes and bypasses on the live capture owner.
 obs_data_set_string(settings,"preset","natural");obs_source_update(source,settings);
 obs_properties_t *presetProps=obs_source_properties(source);
 obs_property_button_clicked(obs_properties_get(presetProps,"applyPreset"),source);
 obs_properties_destroy(presetProps);
 obs_data_t *applied=obs_source_get_settings(source);
 if(obs_data_get_int(applied,"style")!=1 || obs_data_get_double(applied,"tone")!=1)return 16;
 obs_data_release(applied);
 obs_data_set_string(settings,"preset","custom");obs_data_set_int(settings,"style",2);obs_data_set_double(settings,"tone",.5);obs_data_set_double(settings,"structure",.7);obs_data_set_bool(settings,"automaticMask",true);obs_data_set_double(settings,"skinStructure",.4);obs_data_set_double(settings,"maskStructure",.8);obs_data_set_double(settings,"intensity",.65);obs_data_set_int(settings,"noiseFrame",17);
 obs_source_update(source,settings);
 obs_properties_t *reopened=obs_source_properties(source);obs_properties_destroy(reopened);
 obs_data_t *persisted=obs_source_get_settings(source);
 if(obs_data_get_int(persisted,"style")!=2 || obs_data_get_double(persisted,"tone")!=.5)return 15;
 obs_data_release(persisted);pump(8);
 NSData *control=[NSData dataWithContentsOfFile:[output stringByAppendingPathComponent:@"runtime.json"]];
 NSDictionary *custom=[NSJSONSerialization JSONObjectWithData:control options:0 error:nil];
 if([custom[@"failed"] intValue] || [custom[@"settings"][@"style"] intValue]!=2 || ![custom[@"settings"][@"automaticMask"] boolValue])return 14;
 [control writeToFile:[output stringByAppendingPathComponent:@"custom-controls.json"] atomically:YES];
 obs_data_set_bool(settings,"enabled",false);obs_data_set_bool(settings,"upscaling",false);obs_source_update(source,settings);pump(7);
 NSData *bypass=[NSData dataWithContentsOfFile:[output stringByAppendingPathComponent:@"runtime.json"]];
 [bypass writeToFile:[output stringByAppendingPathComponent:@"bypass.json"] atomically:YES];
 NSDictionary *check=[NSJSONSerialization JSONObjectWithData:bypass options:0 error:nil];
 if([check[@"failed"] intValue] || [check[@"stages"] objectForKey:@"pass1MS"])return 12;
 obs_set_output_source(0,NULL);obs_source_release(source);obs_data_release(settings);obs_shutdown();
 NSLog(@"PASS all three OBS live modes, controls and bypass");
 return 0;
}}
