#import "DLSSControls.h"
#import <obs-frontend-api.h>
#import <AVFoundation/AVFoundation.h>

static NSColor *lime(void){return [NSColor colorWithCalibratedRed:.69 green:1 blue:.28 alpha:1];}
static NSColor *panelColor(void){return [NSColor colorWithCalibratedRed:.09 green:.14 blue:.13 alpha:.98];}
static NSTextField *label(NSString *text,CGFloat size,NSFontWeight weight){NSTextField *v=[NSTextField labelWithString:text];v.font=[NSFont systemFontOfSize:size weight:weight];v.textColor=NSColor.whiteColor;[v setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow forOrientation:NSLayoutConstraintOrientationHorizontal];v.preferredMaxLayoutWidth=640;return v;}
static void height(NSView *view,CGFloat value){[view.heightAnchor constraintEqualToConstant:value].active=YES;}
static NSView *line(void){NSView *v=[NSView new];v.wantsLayer=YES;v.layer.backgroundColor=[NSColor colorWithWhite:1 alpha:.14].CGColor;height(v,1);return v;}

@interface DLSSButton : NSButton
@property BOOL selectedStyle;
@property(copy) void (^clicked)(DLSSButton *);
@end
@implementation DLSSButton
- (instancetype)initWithTitle:(NSString *)title {if((self=[super init])){self.title=title;self.target=self;self.action=@selector(invoke:);self.bordered=NO;self.font=[NSFont systemFontOfSize:15 weight:NSFontWeightHeavy];self.focusRingType=NSFocusRingTypeExterior;}return self;}
- (void)invoke:(id)sender{(void)sender;if(self.clicked)self.clicked(self);}
- (void)drawRect:(NSRect)dirty {(void)dirty;NSBezierPath *p=[NSBezierPath bezierPathWithRoundedRect:NSInsetRect(self.bounds,2,2) xRadius:11 yRadius:11];[(self.selectedStyle ? [lime() colorWithAlphaComponent:.18]:[NSColor colorWithWhite:1 alpha:.04]) setFill];[p fill];[(self.selectedStyle ? lime():[NSColor colorWithWhite:.7 alpha:.4]) setStroke];p.lineWidth=self.selectedStyle ? 2.5:1.3;[p stroke];NSDictionary *a=@{NSFontAttributeName:self.font,NSForegroundColorAttributeName:self.selectedStyle ? lime():NSColor.whiteColor};NSSize s=[self.title sizeWithAttributes:a];[self.title drawAtPoint:NSMakePoint((self.bounds.size.width-s.width)/2,(self.bounds.size.height-s.height)/2) withAttributes:a];}
@end

@interface DLSSCheck : NSButton
@property(copy) void (^changed)(DLSSCheck *);
@end
@implementation DLSSCheck
- (instancetype)initWithTitle:(NSString *)title {if((self=[super init])){self.title=title;[self setButtonType:NSButtonTypeSwitch];self.target=self;self.action=@selector(invoke:);self.bordered=NO;self.font=[NSFont systemFontOfSize:17 weight:NSFontWeightHeavy];}return self;}
- (void)invoke:(id)sender{(void)sender;self.needsDisplay=YES;if(self.changed)self.changed(self);}
- (void)drawRect:(NSRect)dirty{(void)dirty;NSRect square=NSMakeRect(2,(self.bounds.size.height-25)/2,25,25);NSBezierPath *box=[NSBezierPath bezierPathWithRoundedRect:square xRadius:4 yRadius:4];[(self.state==NSControlStateValueOn ? lime():[NSColor colorWithCalibratedRed:.22 green:.30 blue:.30 alpha:1]) setFill];[box fill];[[NSColor colorWithWhite:1 alpha:.55] setStroke];box.lineWidth=1;[box stroke];if(self.state==NSControlStateValueOn){NSBezierPath *tick=[NSBezierPath bezierPath];[tick moveToPoint:NSMakePoint(square.origin.x+6,square.origin.y+12)];[tick lineToPoint:NSMakePoint(square.origin.x+11,square.origin.y+18)];[tick lineToPoint:NSMakePoint(square.origin.x+21,square.origin.y+6)];tick.lineWidth=3;[panelColor() setStroke];[tick stroke];}NSDictionary *a=@{NSFontAttributeName:self.font,NSForegroundColorAttributeName:NSColor.whiteColor};NSSize size=[self.title sizeWithAttributes:a];[self.title drawAtPoint:NSMakePoint(39,(self.bounds.size.height-size.height)/2) withAttributes:a];}
@end

@interface DLSSNumberCell : NSTextFieldCell @end
@implementation DLSSNumberCell
- (NSRect)drawingRectForBounds:(NSRect)rect {NSRect r=[super drawingRectForBounds:rect];CGFloat h=ceil(self.font.ascender-self.font.descender)+2;if(r.size.height>h){r.origin.y+=(r.size.height-h)/2;r.size.height=h;}return r;}
@end
@interface DLSSSliderCell : NSSliderCell @end
@implementation DLSSSliderCell
- (CGFloat)knobThickness{return 23;}
- (void)drawBarInside:(NSRect)rect flipped:(BOOL)flipped{(void)flipped;NSRect bar=NSMakeRect(rect.origin.x,NSMidY(rect)-3.5,rect.size.width,7);NSBezierPath *p=[NSBezierPath bezierPathWithRoundedRect:bar xRadius:3.5 yRadius:3.5];[[NSColor colorWithCalibratedRed:.47 green:.57 blue:.58 alpha:1] setFill];[p fill];double ratio=(self.doubleValue-self.minValue)/MAX(.00001,self.maxValue-self.minValue);bar.size.width*=MAX(0,MIN(1,ratio));p=[NSBezierPath bezierPathWithRoundedRect:bar xRadius:3.5 yRadius:3.5];[(self.enabled ? lime():[NSColor colorWithWhite:.5 alpha:1]) setFill];[p fill];}
- (void)drawKnob:(NSRect)rect {NSBezierPath *p=[NSBezierPath bezierPathWithOvalInRect:NSMakeRect(NSMidX(rect)-11,NSMidY(rect)-11,22,22)];[(self.enabled ? lime():[NSColor colorWithWhite:.5 alpha:1]) setFill];[p fill];}
@end
@interface DLSSSlider : NSSlider
@property BOOL tracking;
@end
@implementation DLSSSlider
- (void)mouseDown:(NSEvent *)event{self.tracking=YES;[super mouseDown:event];self.tracking=NO;[NSApp sendAction:self.action to:self.target from:self];}
@end
@interface DLSSScroller : NSScroller @end
@implementation DLSSScroller
- (void)drawKnob{NSRect rect=[self rectForPart:NSScrollerKnob];[lime() setFill];[[NSBezierPath bezierPathWithRoundedRect:NSInsetRect(rect,4,1) xRadius:3 yRadius:3] fill];}
@end
@interface DLSSStack : NSStackView @end
@implementation DLSSStack
- (BOOL)isAccessibilityElement{return YES;}
- (NSString *)accessibilityRole{return NSAccessibilityGroupRole;}
- (NSArray *)accessibilityChildren{return NSAccessibilityUnignoredChildren(self.arrangedSubviews);}
@end
@interface DLSSFlipped : NSView @end
@implementation DLSSFlipped
- (BOOL)isFlipped{return YES;}
- (BOOL)isAccessibilityElement{return YES;}
- (NSString *)accessibilityRole{return NSAccessibilityGroupRole;}
- (NSArray *)accessibilityChildren{return NSAccessibilityUnignoredChildren(self.subviews);}
@end

@interface DLSSControlsController : NSWindowController <NSWindowDelegate,NSTextFieldDelegate>
@property obs_source_t *source;
@property(copy) NSDictionary *(^statusProvider)(void);
@property NSMutableDictionary<NSString *,NSMutableArray<NSControl *> *> *controls;
@property NSMutableArray<DLSSButton *> *styleButtons;
@property NSTimer *timer;
@property NSScrollView *scroll;
@property NSStackView *stack;
@property DLSSButton *controlsTab,*toolsTab;
@property NSTextField *badge,*fps,*statistics;
@property BOOL tools,fetching;
@property double previousTime,lastInputTime;
@property uint64_t previousSelected,previousSubmitted;
@end
@implementation DLSSControlsController
- (instancetype)init {
 NSRect available=NSScreen.mainScreen.visibleFrame;CGFloat h=MIN(960,available.size.height-55);
 NSPanel *window=[[NSPanel alloc] initWithContentRect:NSMakeRect(0,0,760,h) styleMask:NSWindowStyleMaskTitled|NSWindowStyleMaskClosable|NSWindowStyleMaskResizable backing:NSBackingStoreBuffered defer:NO];
 window.title=@"DLSS 5";window.titleVisibility=NSWindowTitleHidden;window.titlebarAppearsTransparent=YES;window.backgroundColor=panelColor();window.minSize=NSMakeSize(680,650);window.maxSize=NSMakeSize(1000,available.size.height);window.releasedWhenClosed=NO;window.hidesOnDeactivate=NO;window.collectionBehavior=NSWindowCollectionBehaviorFullScreenAuxiliary;
 if((self=[super initWithWindow:window])){window.delegate=self;self.controls=[NSMutableDictionary dictionary];self.styleButtons=[NSMutableArray array];[window center];[self build];}return self;
}
- (void)dealloc {[_timer invalidate];if(_source)obs_source_release(_source);}
- (void)attach:(obs_source_t *)source status:(NSDictionary *(^)(void))provider {
 if(_source!=source){if(_source)obs_source_release(_source);_source=obs_source_get_ref(source);}
 self.statusProvider=provider;self.previousTime=0;self.previousSelected=0;self.previousSubmitted=0;self.lastInputTime=NSProcessInfo.processInfo.systemUptime;
 [self rebuild];[_timer invalidate];__weak __typeof__(self) weak=self;self.timer=[NSTimer scheduledTimerWithTimeInterval:1 repeats:YES block:^(NSTimer *t){(void)t;[weak refresh];}];[self refresh];
}
- (void)windowWillClose:(NSNotification *)notification {(void)notification;[_timer invalidate];self.timer=nil;self.statusProvider=nil;if(_source){obs_source_release(_source);_source=NULL;}}
- (NSStackView *)horizontal:(NSArray<NSView *> *)items {NSStackView *v=[DLSSStack new];for(NSView *item in items)[v addArrangedSubview:item];v.orientation=NSUserInterfaceLayoutOrientationHorizontal;v.spacing=14;v.alignment=NSLayoutAttributeCenterY;return v;}
- (void)add:(NSView *)v {[self.stack addArrangedSubview:v];[v.widthAnchor constraintEqualToAnchor:self.stack.widthAnchor].active=YES;}
- (void)build {
 NSView *root=[NSView new];root.wantsLayer=YES;root.layer.backgroundColor=panelColor().CGColor;root.layer.borderColor=lime().CGColor;root.layer.borderWidth=2;root.layer.cornerRadius=20;self.window.contentView=root;
 self.fps=label(@"FPS —",20,NSFontWeightHeavy);self.fps.toolTip=@"Native shown-frame rate when the ProMotion preview is active; otherwise OBS source-selection rate. NR completion FPS is reported separately.";self.fps.translatesAutoresizingMaskIntoConstraints=NO;[root addSubview:self.fps];
 self.scroll=[NSScrollView new];self.scroll.drawsBackground=NO;self.scroll.hasVerticalScroller=YES;self.scroll.verticalScroller=[DLSSScroller new];self.scroll.scrollerStyle=NSScrollerStyleOverlay;self.scroll.translatesAutoresizingMaskIntoConstraints=NO;[root addSubview:self.scroll];
 __weak __typeof__(self) weak=self;self.toolsTab=[[DLSSButton alloc] initWithTitle:@"Live tools"];self.controlsTab=[[DLSSButton alloc] initWithTitle:@"DLSS controls"];
 self.toolsTab.clicked=^(DLSSButton *b){(void)b;DLSSControlsController *s=weak;s.tools=YES;[s rebuild];};self.controlsTab.clicked=^(DLSSButton *b){(void)b;DLSSControlsController *s=weak;s.tools=NO;[s rebuild];};
 NSStackView *tabs=[self horizontal:@[self.toolsTab,self.controlsTab]];tabs.distribution=NSStackViewDistributionFillEqually;tabs.translatesAutoresizingMaskIntoConstraints=NO;[root addSubview:tabs];height(tabs,52);height(self.toolsTab,48);height(self.controlsTab,48);
 [NSLayoutConstraint activateConstraints:@[[self.fps.leadingAnchor constraintEqualToAnchor:root.leadingAnchor constant:27],[self.fps.topAnchor constraintEqualToAnchor:root.topAnchor constant:28],[self.scroll.topAnchor constraintEqualToAnchor:self.fps.bottomAnchor constant:10],[self.scroll.leadingAnchor constraintEqualToAnchor:root.leadingAnchor constant:12],[self.scroll.trailingAnchor constraintEqualToAnchor:root.trailingAnchor constant:-12],[self.scroll.bottomAnchor constraintEqualToAnchor:tabs.topAnchor constant:-14],[tabs.leadingAnchor constraintEqualToAnchor:root.leadingAnchor constant:22],[tabs.trailingAnchor constraintEqualToAnchor:root.trailingAnchor constant:-22],[tabs.bottomAnchor constraintEqualToAnchor:root.bottomAnchor constant:-20]]];
}
- (void)registerControl:(NSControl *)control key:(NSString *)key {control.identifier=key;[control setAccessibilityIdentifier:key];if(!self.controls[key])self.controls[key]=[NSMutableArray array];[self.controls[key] addObject:control];}
- (void)check:(NSString *)title key:(NSString *)key hint:(NSString *)hint {
 DLSSCheck *check=[[DLSSCheck alloc] initWithTitle:title];height(check,43);check.toolTip=hint;[self registerControl:check key:key];__weak __typeof__(self) weak=self;
 check.changed=^(DLSSCheck *sender){[weak setBool:sender.state==NSControlStateValueOn key:key];};
 if(hint.length){NSTextField *note=label(hint,11,NSFontWeightMedium);note.textColor=[NSColor colorWithWhite:.8 alpha:1];note.lineBreakMode=NSLineBreakByWordWrapping;note.maximumNumberOfLines=2;[note.widthAnchor constraintEqualToConstant:235].active=YES;NSStackView *row=[self horizontal:@[check,note]];[check setContentHuggingPriority:100 forOrientation:NSLayoutConstraintOrientationHorizontal];[self add:row];}
 else [self add:check];
}
- (void)setting:(NSString *)title key:(NSString *)key minimum:(double)minimum maximum:(double)maximum value:(double)value supported:(BOOL)supported help:(NSString *)help {
 NSTextField *name=label(title,14,NSFontWeightSemibold);[name.widthAnchor constraintEqualToConstant:205].active=YES;
 DLSSSlider *slider=[DLSSSlider new];slider.cell=[DLSSSliderCell new];slider.minValue=minimum;slider.maxValue=maximum;slider.doubleValue=value;slider.continuous=YES;slider.target=self;slider.action=@selector(sliderChanged:);slider.enabled=supported;slider.toolTip=help;
 NSTextField *number=[NSTextField new];number.cell=[DLSSNumberCell new];number.doubleValue=value;number.alignment=NSTextAlignmentCenter;number.font=[NSFont monospacedDigitSystemFontOfSize:16 weight:NSFontWeightBold];number.textColor=NSColor.whiteColor;number.backgroundColor=[NSColor colorWithCalibratedRed:.20 green:.28 blue:.28 alpha:1];number.drawsBackground=YES;number.bordered=YES;number.bezeled=NO;number.wantsLayer=YES;number.layer.cornerRadius=7;number.layer.borderWidth=1;number.layer.borderColor=[NSColor colorWithWhite:.7 alpha:.6].CGColor;number.enabled=supported;number.toolTip=help;number.delegate=self;[number setAccessibilityLabel:[title stringByAppendingString:@" value"]];[slider setAccessibilityLabel:title];
 number.editable=supported;number.selectable=supported;
 NSNumberFormatter *format=[NSNumberFormatter new];format.minimum=@(minimum);format.maximum=@(maximum);format.minimumFractionDigits=2;format.maximumFractionDigits=2;format.numberStyle=NSNumberFormatterDecimalStyle;number.formatter=format;
 [number.widthAnchor constraintEqualToConstant:86].active=YES;height(number,37);height(slider,35);
 [self registerControl:slider key:key];[self registerControl:number key:key];NSStackView *row=[self horizontal:@[name,slider,number]];height(row,52);if(!supported)row.alphaValue=.46;[self add:row];
}
- (void)section:(NSString *)title {[self add:label(title,20,NSFontWeightHeavy)];}
- (void)rebuild {
 [self.controls removeAllObjects];[self.styleButtons removeAllObjects];self.statistics=nil;
 DLSSFlipped *document=[DLSSFlipped new];document.translatesAutoresizingMaskIntoConstraints=NO;self.stack=[DLSSStack new];self.stack.orientation=NSUserInterfaceLayoutOrientationVertical;self.stack.alignment=NSLayoutAttributeLeading;self.stack.spacing=21;self.stack.translatesAutoresizingMaskIntoConstraints=NO;[document addSubview:self.stack];self.scroll.documentView=document;
 [NSLayoutConstraint activateConstraints:@[[document.widthAnchor constraintEqualToAnchor:self.scroll.contentView.widthAnchor],[self.stack.leadingAnchor constraintEqualToAnchor:document.leadingAnchor constant:29],[self.stack.trailingAnchor constraintEqualToAnchor:document.trailingAnchor constant:-29],[self.stack.topAnchor constraintEqualToAnchor:document.topAnchor constant:22],[self.stack.bottomAnchor constraintEqualToAnchor:document.bottomAnchor constant:-28]]];
 self.controlsTab.selectedStyle=!self.tools;self.controlsTab.needsDisplay=YES;self.toolsTab.selectedStyle=self.tools;self.toolsTab.needsDisplay=YES;
 NSTextField *title=label(self.tools ? @"DLSS 5 LIVE TOOLS":@"DLSS 5 SWAPPER CONTROLS",21,NSFontWeightHeavy);self.badge=label(@"CONNECTING",12,NSFontWeightHeavy);self.badge.textColor=lime();self.badge.alignment=NSTextAlignmentCenter;self.badge.wantsLayer=YES;self.badge.layer.borderColor=lime().CGColor;self.badge.layer.borderWidth=1.5;self.badge.layer.cornerRadius=7;[self.badge.widthAnchor constraintEqualToConstant:120].active=YES;height(self.badge,28);NSView *space=[NSView new];[self add:[self horizontal:@[title,space,self.badge]]];
 if(self.tools){[self buildTools];}else{
 [self check:@"DLSS ON" key:@"enabled" hint:@"OBS ·live NR pipeline"];
 [self check:@"ON-SCREEN STATUS" key:@"onScreenStatus" hint:@"Show DLSS ON/OFF over the game"];
 [self section:@"GLOBAL CONTROLS"];
 [self setting:@"Structure Intensity" key:@"structure" minimum:0 maximum:2 value:1 supported:YES help:@"Recovered NR local-structure conditioning input."];
 [self setting:@"Tone Intensity" key:@"tone" minimum:0 maximum:2 value:1 supported:YES help:@"Recovered NR tone input; linked to Local Tone Intensity below."];
 [self add:line()];[self check:@"CHARACTER MASK" key:@"automaticMask" hint:@"NR skin/mask conditioning"];
 [self setting:@"Character/Skin Structure" key:@"skinStructure" minimum:-1 maximum:2 value:-1 supported:YES help:@"Model conditioning, not an external character-ID mask. −1 inherits Structure Intensity."];
 [self add:line()];[self section:@"NR STYLE"];
 NSMutableArray *buttons=[NSMutableArray array];NSArray *names=@[@"Model A ·Default",@"Model B ·Natural",@"Model C ·Cinematic"];__weak __typeof__(self) weak=self;
 for(NSInteger i=0;i<3;i++){DLSSButton *b=[[DLSSButton alloc] initWithTitle:names[i]];height(b,62);b.clicked=^(DLSSButton *button){(void)button;[weak setStyle:i];};[self.styleButtons addObject:b];[buttons addObject:b];}
 NSStackView *styles=[self horizontal:buttons];styles.distribution=NSStackViewDistributionFillEqually;[self add:styles];
 [self add:line()];[self section:@"MORE RENODX CONTROLS"];
 [self popup:@"Intensity Mode" key:@"intensityMode" labels:@[@"Final",@"Per-Layer"] values:@[@"final",@"layer"]];
 [self setting:@"Final Intensity" key:@"intensity" minimum:0 maximum:2 value:1 supported:YES help:@"Blends or amplifies the finished result against the source once."];
 [self setting:@"Per-Layer Intensity" key:@"layerIntensity" minimum:0 maximum:2 value:1 supported:YES help:@"Scales the current prediction in each NR pass. In two-pass modes the next model receives the adjusted result."];
 [self setting:@"Local Tone Intensity" key:@"tone" minimum:0 maximum:2 value:1 supported:YES help:@"Linked to Tone Intensity: this backend exposes one recovered tone-conditioning input."];
 [self setting:@"SDR Diffuse White" key:@"diffuseWhite" minimum:80 maximum:400 value:80 supported:YES help:@"SDR white reference, relative to 80 nits. Output remains SDR."];
 [self setting:@"Luminance Transfer" key:@"hdrStrength" minimum:0 maximum:2 value:1 supported:YES help:@"Strength of NR luminance change in screen space."];
 [self setting:@"Color Transfer" key:@"colorStrength" minimum:0 maximum:2 value:1 supported:YES help:@"Strength of NR chroma change in screen space."];
 [self check:@"PROTECT HUD RECTANGLE" key:@"uiCorrection" hint:@"Set the rectangle in original OBS source properties"];
 [self check:@"TEMPORAL NR" key:@"temporalNR" hint:@"Use motion estimated from the HDMI image"];
 [self setting:@"Motion Scale X Multiplier" key:@"motionScaleX" minimum:-2 maximum:2 value:1 supported:YES help:@"Horizontal scale for estimated screen-space NR motion."];
 [self setting:@"Motion Scale Y Multiplier" key:@"motionScaleY" minimum:-2 maximum:2 value:1 supported:YES help:@"Vertical scale for estimated screen-space NR motion."];
 [self add:line()];NSTextField *note=label(@"Live NR settings. A/B/C select NR style, not separate AI models. Click a number to type. Tone controls share the same recovered input. Temporal NR uses estimated screen-space motion. Live tools keeps the original OBS controls available.",11,NSFontWeightMedium);note.maximumNumberOfLines=0;note.lineBreakMode=NSLineBreakByWordWrapping;note.textColor=[NSColor colorWithWhite:.8 alpha:1];[self add:note];
 }
 [self refreshControls];[self.scroll.contentView scrollToPoint:NSZeroPoint];[self.scroll reflectScrolledClipView:self.scroll.contentView];
}
- (void)popup:(NSString *)title key:(NSString *)key labels:(NSArray<NSString *> *)labels values:(NSArray *)values {
 NSTextField *name=label(title,14,NSFontWeightSemibold);[name.widthAnchor constraintEqualToConstant:180].active=YES;NSPopUpButton *popup=[[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
 for(NSUInteger i=0;i<labels.count;i++){[popup addItemWithTitle:labels[i]];popup.lastItem.representedObject=values[i];}popup.target=self;popup.action=@selector(popupChanged:);[self registerControl:popup key:key];[self add:[self horizontal:@[name,popup]]];
}
- (void)buildTools {
 NSMutableArray *deviceNames=[NSMutableArray arrayWithObject:@"Select HDMI capture device"],*deviceIDs=[NSMutableArray arrayWithObject:@""];
 AVCaptureDeviceDiscoverySession *discovery=[AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeExternal] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
 for(AVCaptureDevice *device in discovery.devices){[deviceNames addObject:device.localizedName];[deviceIDs addObject:device.uniqueID];}
 [self popup:@"HDMI capture device" key:@"device" labels:deviceNames values:deviceIDs];
 __weak __typeof__(self) weakCamera=self;
 DLSSButton *authorize=[[DLSSButton alloc] initWithTitle:@"Camera permission / connect HDMI"];height(authorize,44);
 authorize.clicked=^(DLSSButton *button){(void)button;DLSSControlsController *s=weakCamera;if(!s.source)return;obs_source_t *lease=obs_source_get_ref(s.source);
  [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted){(void)granted;dispatch_async(dispatch_get_main_queue(),^{obs_data_t *s=obs_source_get_settings(lease);obs_source_update(lease,s);obs_data_release(s);obs_source_release(lease);});}];};[self add:authorize];
 [self section:@"NEURAL RENDERING"];
 [self popup:@"DLSS mode" key:@"mode" labels:@[@"DLSS Fidelity ·71 →71",@"DLSS Balanced ·31 →71",@"DLSS Performance ·31"] values:@[@"71x2-480",@"31then71-480",@"31-480"]];
 [self check:@"LEARNED UPSCALING" key:@"upscaling" hint:@"Existing reconstruction branch"];
 [self add:line()];[self section:@"MULTI-FRAME GENERATION"];
 [self popup:@"Interpolation engine" key:@"frameGenerationBackend" labels:@[@"MLX AI frame generation",@"Apple native frame generation"] values:@[@"mlx",@"apple"]];
 [self popup:@"Maximum factor" key:@"frameGenerationFactor" labels:@[@"Off",@"2×",@"3×",@"4×",@"5×",@"6×"] values:@[@1,@2,@3,@4,@5,@6]];
 [self check:@"CAPTURE-GUIDED MOTION" key:@"captureGuided" hint:@"Use actual HDMI intermediate geometry"];
 [self popup:@"MLX motion detail" key:@"frameGenerationHeight" labels:@[@"Fast ·360p",@"Detailed ·540p",@"High ·720p"] values:@[@360,@540,@720]];
 NSTextField *hint=label(@"Capture-guided mode reconstructs appearance on real intermediate HDMI frames. The maximum factor is a cap; it does not invent duplicate capture timestamps or guarantee a display rate. Apple motion processing uses full1080p.",12,NSFontWeightMedium);hint.maximumNumberOfLines=0;hint.lineBreakMode=NSLineBreakByWordWrapping;[self add:hint];
 __weak __typeof__(self) weak=self;
 DLSSButton *preview=[[DLSSButton alloc] initWithTitle:@"Open native ProMotion preview"];height(preview,48);preview.clicked=^(DLSSButton *b){(void)b;DLSSControlsController *s=weak;if(s.source){obs_properties_t *p=obs_source_properties(s.source);obs_property_t *button=obs_properties_get(p,"preview");if(button)obs_property_button_clicked(button,s.source);obs_properties_destroy(p);}};[self add:preview];
 DLSSButton *advanced=[[DLSSButton alloc] initWithTitle:@"Original OBS source properties"];height(advanced,48);advanced.clicked=^(DLSSButton *b){(void)b;DLSSControlsController *s=weak;if(s.source)obs_frontend_open_source_properties(s.source);};[self add:advanced];
 [self add:line()];[self section:@"LIVE MEASUREMENTS"];self.statistics=label(@"Waiting for measurements…",12,NSFontWeightMedium);self.statistics.maximumNumberOfLines=0;self.statistics.lineBreakMode=NSLineBreakByWordWrapping;[self add:self.statistics];
}
- (void)setBool:(BOOL)value key:(NSString *)key {if(!_source)return;obs_data_t *s=obs_source_get_settings(_source);obs_data_set_bool(s,key.UTF8String,value);obs_source_update(_source,s);obs_data_release(s);[self refreshControls];}
- (void)setStyle:(NSInteger)style {if(!_source)return;obs_data_t *s=obs_source_get_settings(_source);obs_data_set_int(s,"style",style);obs_source_update(_source,s);obs_data_release(s);[self refreshControls];}
- (void)setNumber:(double)value key:(NSString *)key {if(!_source || !isfinite(value))return;obs_data_t *s=obs_source_get_settings(_source);obs_data_set_double(s,key.UTF8String,value);obs_source_update(_source,s);obs_data_release(s);[self refreshControls];}
- (void)sliderChanged:(DLSSSlider *)sender {for(NSControl *c in self.controls[sender.identifier])if(c!=sender && [c isKindOfClass:NSTextField.class])[(NSTextField *)c setDoubleValue:sender.doubleValue];if(!sender.tracking)[self setNumber:sender.doubleValue key:sender.identifier];}
- (void)controlTextDidEndEditing:(NSNotification *)note {NSTextField *f=note.object;if(f.enabled && ![f.identifier hasPrefix:@"unavailable"])[self setNumber:f.doubleValue key:f.identifier];}
- (void)popupChanged:(NSPopUpButton *)sender {if(!_source)return;id value=sender.selectedItem.representedObject;obs_data_t *s=obs_source_get_settings(_source);if([value isKindOfClass:NSString.class])obs_data_set_string(s,sender.identifier.UTF8String,[value UTF8String]);else obs_data_set_int(s,sender.identifier.UTF8String,[value longLongValue]);obs_source_update(_source,s);obs_data_release(s);[self refreshControls];}
- (void)refreshControls {
 if(!_source)return;obs_data_t *s=obs_source_get_settings(_source);
 for(NSString *key in self.controls){if([key hasPrefix:@"unavailable"])continue;BOOL tracking=NO;for(NSControl *candidate in self.controls[key])if([candidate isKindOfClass:DLSSSlider.class] && [(DLSSSlider *)candidate tracking])tracking=YES;if(tracking)continue;for(NSControl *c in self.controls[key]){
  if([c isKindOfClass:DLSSCheck.class]){[(DLSSCheck *)c setState:obs_data_get_bool(s,key.UTF8String) ? NSControlStateValueOn:NSControlStateValueOff];c.needsDisplay=YES;}
  else if([c isKindOfClass:NSPopUpButton.class]){NSPopUpButton *p=(id)c;for(NSMenuItem *item in p.itemArray){id v=item.representedObject;BOOL equal=[v isKindOfClass:NSString.class] ? [v isEqual:@(obs_data_get_string(s,key.UTF8String))] : [v longLongValue]==obs_data_get_int(s,key.UTF8String);if(equal){[p selectItem:item];break;}}}
  else {if([c isKindOfClass:DLSSSlider.class] && [(DLSSSlider *)c tracking])continue;if([c isKindOfClass:NSTextField.class] && [(NSTextField *)c currentEditor])continue;c.doubleValue=obs_data_get_double(s,key.UTF8String);}
 }}
 for(NSUInteger i=0;i<self.styleButtons.count;i++){self.styleButtons[i].selectedStyle=obs_data_get_int(s,"style")== (long long)i;self.styleButtons[i].needsDisplay=YES;}
 for(NSControl *c in self.controls[@"skinStructure"])c.enabled=obs_data_get_bool(s,"automaticMask");
 for(NSControl *c in self.controls[@"frameGenerationHeight"])c.enabled=!strcmp(obs_data_get_string(s,"frameGenerationBackend"),"mlx");
 BOOL layerMode=!strcmp(obs_data_get_string(s,"intensityMode"),"layer");
 for(NSControl *c in self.controls[@"intensity"])c.enabled=!layerMode;
 for(NSControl *c in self.controls[@"layerIntensity"])c.enabled=layerMode;
 obs_data_release(s);
}
- (void)refresh {
 [self refreshControls];if(self.fetching || !self.statusProvider || !_source)return;self.fetching=YES;NSDictionary *(^provider)(void)=[self.statusProvider copy];__weak __typeof__(self) weak=self;
 dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY,0),^{NSDictionary *status=provider();dispatch_async(dispatch_get_main_queue(),^{DLSSControlsController *s=weak;if(!s)return;s.fetching=NO;if(!s.source || !s.window.visible)return;[s applyStatus:status];});});
}
- (void)applyStatus:(NSDictionary *)s {
 double now=[s[@"sampledAtHostSeconds"] doubleValue];uint64_t selected=[s[@"obsRenderedUnique"] unsignedLongLongValue],submitted=[s[@"submitted"] unsignedLongLongValue];
 if(submitted>self.previousSubmitted)self.lastInputTime=now;self.previousSubmitted=submitted;
 BOOL connected=[s[@"captureStarts"] intValue]>0 && [s[@"captureError"] length]==0 && now-self.lastInputTime<3;
 self.badge.stringValue=connected ? @"CONNECTED":([s[@"captureAuthorization"] intValue]!=3 ? @"CAMERA ACCESS":@"NO HDMI FEED");self.badge.textColor=connected ? lime():NSColor.systemOrangeColor;
 NSDictionary *native=s[@"nativePreview"];double fps=-1;if([native[@"active"] boolValue])fps=[native[@"uniqueShownFPS"] doubleValue];else if(self.previousTime>0 && now>self.previousTime && selected>=self.previousSelected)fps=(selected-self.previousSelected)/(now-self.previousTime);
 self.fps.stringValue=fps>=0 ? [NSString stringWithFormat:@"FPS %.0f   ·   NR %.1f",fps,[s[@"processedFPS"] doubleValue]]:[NSString stringWithFormat:@"FPS —   ·   NR %.1f",[s[@"processedFPS"] doubleValue]];self.previousTime=now;self.previousSelected=selected;
 if(self.statistics)self.statistics.stringValue=[NSString stringWithFormat:@"%@\nNR: %.1f completed frames/s ·%.1f ms median\nMFG: %@× ·%@ intermediates ·%@ capture-guided\nOBS selected: %@ base /%@ intermediates /%@ repeats\nNative processor: %@ ·%@ skipped requests\nCapture: %@ starts ·%@ drops ·%@\nErrors: NR %@ /FG %@\n%@%@",s[@"backend"] ?: @"",[s[@"processedFPS"] doubleValue],[s[@"processingMedianMS"] doubleValue],s[@"mfgFactor"] ?: @1,s[@"generatedFrames"] ?: @0,s[@"guidedFrames"] ?: @0,s[@"obsSelectedBase"] ?: @0,s[@"obsSelectedGenerated"] ?: @0,s[@"obsRepeatedDraws"] ?: @0,s[@"nativeFGState"] ?: @"inactive",s[@"nativeFGSkippedPairs"] ?: @0,s[@"captureStarts"] ?: @0,s[@"captureDrops"] ?: @0,connected ? @"connected":@"unavailable",s[@"failed"] ?: @0,s[@"mfgFailures"] ?: @0,s[@"captureError"] ?: @"",s[@"mfgError"] ?: @""];
}
@end

static DLSSControlsController *controller;
void dlss_controls_close(void) {
 if(controller)[controller close];
}
void dlss_controls_show(obs_source_t *source,NSDictionary *(^statusProvider)(void)) {
 NSCAssert(NSThread.isMainThread,@"DLSS controls must open on the main thread");
 if(!controller)controller=[DLSSControlsController new];[controller attach:source status:statusProvider];[controller showWindow:nil];[NSApp activate];[controller.window makeKeyAndOrderFront:nil];
}
