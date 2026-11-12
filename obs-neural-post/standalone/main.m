#import <AppKit/AppKit.h>
#import <AVFoundation/AVFoundation.h>
#import "NativeInput.h"
#import "../nr/dlss-nr.h"

@interface NativeHost : NSObject <NSApplicationDelegate>
@property void *engine;
@property NativeInput *input;
@property NSWindow *controls;
@property NSStatusItem *statusItem;
@property NSPopUpButton *mode,*display,*camera,*quality;
@property NSSlider *intensity;
@property NSPopUpButton *intensityMode,*fgFactor;
@property NSTextField *statusText;
@property NSButton *nrEnabled,*metrics;
@property NSTextView *json;
@property NSMutableDictionary *settings;
@property NSTimer *timer;
@property uint64_t sequence;
@property BOOL active,shown;
@end
@implementation NativeHost
- (NSString *)settingsPath {
 NSString *dir=[NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/NVDLSS5"];
 [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
 return [dir stringByAppendingPathComponent:@"settings.json"];
}
- (void)applicationDidFinishLaunching:(NSNotification *)note {
 (void)note;
 self.settings=[[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:self.settingsPath] ?: [NSData data] options:NSJSONReadingMutableContainers error:nil] mutableCopy];
 if(!self.settings)self.settings=[@{@"mode":@"71x2-480",@"enabled":@YES,@"upscaling":@YES,@"intensityMode":@"final",@"intensity":@1,@"layerIntensity":@1,@"style":@0,@"tone":@1,@"structure":@1,@"skinStructure":@(-1),@"maskStructure":@(-1),@"automaticMask":@NO,@"noiseFrame":@0,@"temporalNR":@YES,@"nativeProcessingHeight":@2160,@"captureGuided":@NO,@"frameGenerationBackend":@"rife",@"frameGenerationFactor":@6,@"frameGenerationHeight":@720} mutableCopy];
 NSString *models=[[NSProcessInfo processInfo] environment][@"NVDLSS_MODEL_DIR"] ?: [NSBundle.mainBundle.resourcePath stringByAppendingPathComponent:@"Models"];
 for(NSString *key in @[@"model71",@"model31",@"srPath",@"frameGenerationPath",@"rifeModelPath"]){
  NSString *file=@{@"model71":@"NeuralRendering.dlssmodel",@"model31":@"NeuralRendering-Small.dlssmodel",@"srPath":@"rt4ksr-x3.safetensors",@"frameGenerationPath":@"framegen.safetensors",@"rifeModelPath":@"rife-v4.25.rmw"}[key];
  if(!self.settings[key])self.settings[key]=[models stringByAppendingPathComponent:file];
 }
 self.settings[@"rifeModelPath"]=[NSBundle.mainBundle.resourcePath stringByAppendingPathComponent:@"rife-v4.25.rmw"];
 self.statusItem=[NSStatusBar.systemStatusBar statusItemWithLength:NSVariableStatusItemLength];self.statusItem.button.title=@"DLSS 5";
 NSMenu *menu=[NSMenu new];
 for(NSArray *entry in @[@[@"Controls",@"showControls"],@[@"Start",@"start"],@[@"Stop filter / output",@"stop"],@[@"Quit",@"quit"]]){
  NSMenuItem *item=[[NSMenuItem alloc] initWithTitle:entry[0] action:NSSelectorFromString(entry[1]) keyEquivalent:@""];item.target=self;[menu addItem:item];
 }self.statusItem.menu=menu;
 self.controls=[[NSWindow alloc] initWithContentRect:NSMakeRect(150,75,650,850) styleMask:NSWindowStyleMaskTitled|NSWindowStyleMaskClosable|NSWindowStyleMaskMiniaturizable backing:NSBackingStoreBuffered defer:NO];
 self.controls.title=@"NVDLSS 5 · Native controls";self.controls.releasedWhenClosed=NO;self.controls.level=NSStatusWindowLevel;
 NSStackView *stack=[NSStackView new];stack.orientation=NSUserInterfaceLayoutOrientationVertical;stack.spacing=10;stack.edgeInsets=NSEdgeInsetsMake(16,16,16,16);stack.frame=self.controls.contentView.bounds;stack.autoresizingMask=NSViewWidthSizable|NSViewHeightSizable;self.controls.contentView=stack;
 [stack addArrangedSubview:[NSTextField labelWithString:@"Native MLX / Metal · No OBS runtime"]];
 self.mode=[self popup:@[@"Mac screen filter",@"External HDMI · ProMotion window"] identifier:@"inputMode" stack:stack];
 self.display=[self popup:@[@"Built-in / primary display"] identifier:@"display" stack:stack];self.display.lastItem.representedObject=@0;
 for(NSScreen *screen in NSScreen.screens){[self.display addItemWithTitle:screen.localizedName];self.display.lastItem.representedObject=screen.deviceDescription[@"NSScreenNumber"];}
 self.camera=[self popup:@[@"Select HDMI capture card"] identifier:@"camera" stack:stack];self.camera.lastItem.representedObject=@"";
 AVCaptureDeviceDiscoverySession *devices=[AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeExternal] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
 for(AVCaptureDevice *device in devices.devices){[self.camera addItemWithTitle:device.localizedName];self.camera.lastItem.representedObject=device.uniqueID;}
 NSArray *qualities=@[@"Fidelity ·71 →71",@"Balanced ·31 →71",@"Performance ·31",@"Single 71",@"Double 31"];
 self.quality=[self popup:qualities identifier:@"quality" stack:stack];
 NSArray *values=@[@"71x2-480",@"31then71-480",@"31-480",@"71-480",@"31x2-480"];
 for(NSUInteger i=0;i<values.count;i++){[self.quality itemAtIndex:i].representedObject=values[i];if([values[i] isEqual:self.settings[@"mode"]])[self.quality selectItemAtIndex:i];}
 self.nrEnabled=[NSButton checkboxWithTitle:@"Neural rendering enabled" target:self action:@selector(updateControls)];self.nrEnabled.state=[self.settings[@"enabled"] boolValue];self.nrEnabled.accessibilityIdentifier=@"nrEnabled";[stack addArrangedSubview:self.nrEnabled];
 self.metrics=[NSButton checkboxWithTitle:@"Performance overlay" target:self action:@selector(metricsChanged)];
 self.metrics.state=[NSUserDefaults.standardUserDefaults objectForKey:@"DLSS.NativeStatsVisible"] ? [NSUserDefaults.standardUserDefaults boolForKey:@"DLSS.NativeStatsVisible"] : NSControlStateValueOn;[stack addArrangedSubview:self.metrics];
 self.intensityMode=[self popup:@[@"Final intensity",@"Per-layer intensity"] identifier:@"intensityMode" stack:stack];
 [self.intensityMode selectItemAtIndex:[self.settings[@"intensityMode"] isEqual:@"layer"] ? 1:0];
 self.intensityMode.target=self;self.intensityMode.action=@selector(intensityModeChanged);
 self.intensity=[NSSlider sliderWithValue:[self.settings[self.intensityMode.indexOfSelectedItem ? @"layerIntensity":@"intensity"] doubleValue] minValue:0 maxValue:2 target:self action:@selector(intensityChanged)];
 self.intensity.continuous=NO;self.intensity.accessibilityIdentifier=@"intensity";[self.intensity.widthAnchor constraintEqualToConstant:600].active=YES;[stack addArrangedSubview:self.intensity];
 self.fgFactor=[self popup:@[@"Frame generation off",@"2× frame generation",@"3× frame generation",@"4× frame generation",@"5× frame generation",@"6× frame generation"] identifier:@"fgFactor" stack:stack];
 [self.fgFactor selectItemAtIndex:MAX(0,MIN(5,[self.settings[@"frameGenerationFactor"] integerValue]-1))];self.fgFactor.target=self;self.fgFactor.action=@selector(fgChanged);
 self.json=[NSTextView new];self.json.font=[NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
 self.json.string=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:self.settings options:NSJSONWritingPrettyPrinted error:nil] encoding:NSUTF8StringEncoding];
 NSScrollView *scroll=[NSScrollView new];scroll.documentView=self.json;scroll.hasVerticalScroller=YES;[scroll.heightAnchor constraintEqualToConstant:220].active=YES;[scroll.widthAnchor constraintEqualToConstant:600].active=YES;[stack addArrangedSubview:scroll];
 NSTextField *help=[NSTextField wrappingLabelWithString:@"Full model parameters are editable above, including mutually exclusive final/per-layer intensity. Screen output passes clicks through; use the DLSS 5 menu to stop. This app's windows are excluded from screen input."];[help.widthAnchor constraintEqualToConstant:600].active=YES;[stack addArrangedSubview:help];
 NSStackView *buttons=[NSStackView new];
 for(NSArray *entry in @[@[@"Start / apply",@"start"],@[@"Apply JSON",@"applyJSON"],@[@"Stop",@"stop"],@[@"Screen permission",@"screenPermission"],@[@"Models",@"chooseModels"],@[@"Quit",@"quit"]]){NSButton *b=[NSButton buttonWithTitle:entry[0] target:self action:NSSelectorFromString(entry[1])];[buttons addArrangedSubview:b];}[stack addArrangedSubview:buttons];
 self.statusText=[NSTextField wrappingLabelWithString:@"Ready. Choose a mode and start."];[self.statusText.widthAnchor constraintEqualToConstant:600].active=YES;[stack addArrangedSubview:self.statusText];
 [self showControls];__weak NativeHost *weak=self;
 self.timer=[NSTimer scheduledTimerWithTimeInterval:1 repeats:YES block:^(NSTimer *t){(void)t;[weak tick];}];
}
- (NSPopUpButton *)popup:(NSArray *)items identifier:(NSString *)identifier stack:(NSStackView *)stack {
 NSPopUpButton *p=[[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];[p addItemsWithTitles:items];p.identifier=identifier;p.accessibilityIdentifier=identifier;[p.widthAnchor constraintEqualToConstant:600].active=YES;[stack addArrangedSubview:p];return p;
}
- (void)showControls {[NSApp activateIgnoringOtherApps:YES];[self.controls makeKeyAndOrderFront:nil];}
- (void)screenPermission {CGRequestScreenCaptureAccess();}
- (void)chooseModels {
 NSOpenPanel *panel=[NSOpenPanel openPanel];panel.canChooseDirectories=YES;panel.canChooseFiles=NO;panel.allowsMultipleSelection=NO;
 if([panel runModal]!=NSModalResponseOK)return;
 for(NSString *key in @[@"model71",@"model31",@"srPath",@"frameGenerationPath"]){
  NSString *name=@{@"model71":@"NeuralRendering.dlssmodel",@"model31":@"NeuralRendering-Small.dlssmodel",@"srPath":@"rt4ksr-x3.safetensors",@"frameGenerationPath":@"framegen.safetensors"}[key];
  self.settings[key]=[panel.URL.path stringByAppendingPathComponent:name];
 }
 [self applyRuntimeSettings];self.statusText.stringValue=@"Model directory saved. Start to load the selected mode.";
}
- (void)updateControls {self.settings[@"enabled"]=self.nrEnabled.state==NSControlStateValueOn ? @YES:@NO;[self applyRuntimeSettings];}
- (void)applyRuntimeSettings {
 NSString *text=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:self.settings options:0 error:nil] encoding:NSUTF8StringEncoding];
 if(self.engine)nr_update(self.engine,text.UTF8String);
 self.json.string=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:self.settings options:NSJSONWritingPrettyPrinted error:nil] encoding:NSUTF8StringEncoding];[self save];
}
- (void)intensityModeChanged {
 self.settings[@"intensityMode"]=self.intensityMode.indexOfSelectedItem ? @"layer":@"final";
 self.intensity.doubleValue=[self.settings[self.intensityMode.indexOfSelectedItem ? @"layerIntensity":@"intensity"] doubleValue];[self applyRuntimeSettings];
}
- (void)intensityChanged {self.settings[self.intensityMode.indexOfSelectedItem ? @"layerIntensity":@"intensity"]=@(self.intensity.doubleValue);[self applyRuntimeSettings];}
- (void)metricsChanged {if(self.engine)nr_set_metrics_overlay(self.engine,self.metrics.state==NSControlStateValueOn);}
- (void)fgChanged {self.settings[@"frameGenerationFactor"]=@(self.fgFactor.indexOfSelectedItem+1);[self applyRuntimeSettings];}
- (void)save {NSData *data=[NSJSONSerialization dataWithJSONObject:self.settings options:NSJSONWritingPrettyPrinted error:nil];[data writeToFile:self.settingsPath atomically:YES];}
- (void)applyJSON {
 NSError *error=nil;id d=[NSJSONSerialization JSONObjectWithData:[self.json.string dataUsingEncoding:NSUTF8StringEncoding] options:NSJSONReadingMutableContainers error:&error];
 if(![d isKindOfClass:NSMutableDictionary.class]){self.statusText.stringValue=error.localizedDescription ?: @"Expected a JSON settings object";return;}
 NSString *text=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:d options:0 error:nil] encoding:NSUTF8StringEncoding];
 if(self.engine&&!nr_update(self.engine,text.UTF8String)){self.statusText.stringValue=@"Model settings failed validation";return;}
 self.settings=d;self.nrEnabled.state=[d[@"enabled"] boolValue];
 [self.intensityMode selectItemAtIndex:[d[@"intensityMode"] isEqual:@"layer"] ? 1:0];
 self.intensity.doubleValue=[d[self.intensityMode.indexOfSelectedItem ? @"layerIntensity":@"intensity"] doubleValue];
 [self.fgFactor selectItemAtIndex:MAX(0,MIN(5,[d[@"frameGenerationFactor"] integerValue]-1))];
 for(NSMenuItem *item in self.quality.itemArray)if([item.representedObject isEqual:d[@"mode"]])[self.quality selectItem:item];
 [self save];self.statusText.stringValue=@"Settings applied";
}
- (void)start {
 [self stop];[self updateControls];self.settings[@"mode"]=self.quality.selectedItem.representedObject;
 NSString *text=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:self.settings options:0 error:nil] encoding:NSUTF8StringEncoding];
 if(!self.engine)self.engine=nr_create(text.UTF8String);else if(!nr_update(self.engine,text.UTF8String)){self.statusText.stringValue=@"Invalid model settings";return;}
 if(!self.engine){self.statusText.stringValue=@"Cannot create model runtime; check asset paths";return;}
 nr_reset_input(self.engine);[self metricsChanged];__weak NativeHost *weak=self;
 self.input=[[NativeInput alloc] initWithFrame:^(CVPixelBufferRef pixels,double pts){NativeHost *s=weak;if(s&&s.engine)nr_submit(s.engine,(void *)pixels,++s->_sequence,pts);}];
 self.active=YES;self.shown=NO;
 if(self.mode.indexOfSelectedItem==0)[self.input startScreen:[self.display.selectedItem.representedObject unsignedIntValue]];
 else {
  if([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo]==AVAuthorizationStatusNotDetermined){
   [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted){dispatch_async(dispatch_get_main_queue(),^{if(granted)[weak start];});}];return;
  }
  [self.input startHDMI:self.camera.selectedItem.representedObject];
 }
 self.json.string=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:self.settings options:NSJSONWritingPrettyPrinted error:nil] encoding:NSUTF8StringEncoding];[self save];
}
- (void)stop {[self.input stop];self.input=nil;self.active=NO;self.shown=NO;if(self.engine){nr_hide_output(self.engine);nr_reset_input(self.engine);}}
- (void)tick {
 if(!self.engine)return;
 char text[262144]={0};nr_status(self.engine,text,sizeof(text));
 NSMutableDictionary *d=[[NSJSONSerialization JSONObjectWithData:[[@(text) description] dataUsingEncoding:NSUTF8StringEncoding] options:NSJSONReadingMutableContainers error:nil] mutableCopy] ?: [NSMutableDictionary dictionary];
 d[@"capture"]=[self.input status] ?: @{};d[@"host"]=@"native-no-obs";d[@"active"]=@(self.active);
 NSString *dir=NSProcessInfo.processInfo.environment[@"NVDLSS_DIAGNOSTICS_DIR"];
 if(dir.length){[[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];[[NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingPrettyPrinted error:nil] writeToFile:[dir stringByAppendingPathComponent:@"runtime.json"] atomically:YES];}
 self.statusText.stringValue=[NSString stringWithFormat:@"%@ · %.1f base processed FPS\n%@",self.active ? @"Running":@"Stopped",[d[@"processedFPS"] doubleValue],self.input.error.length ? self.input.error : (d[@"error"] ?: @"")];
 if(self.active&&!self.shown){void *frame=nr_frame_acquire(self.engine);if(frame){nr_frame_release(frame);self.shown=YES;
  if([[self.input status][@"input"] isEqual:@"screen"])nr_show_screen_filter(self.engine,[[self.input status][@"displayID"] unsignedIntValue]);else nr_show_preview(self.engine);
 }}
}
- (void)quit {[NSApp terminate:nil];}
- (void)applicationWillTerminate:(NSNotification *)note {(void)note;NSLog(@"Native host termination requested");[self stop];if(self.engine){nr_destroy(self.engine);self.engine=NULL;}[self.timer invalidate];}
@end
int main(int argc,const char **argv){(void)argc;(void)argv;@autoreleasepool {NSApplication *app=NSApplication.sharedApplication;[app setActivationPolicy:NSApplicationActivationPolicyAccessory];__attribute__((objc_precise_lifetime)) NativeHost *host=[NativeHost new];app.delegate=host;[app run];}return 0;}
