// OBS native SDR HDMI source. IOSurface handoff follows mac-avcapture's fast path.
// GPL-2.0-or-later, matching OBS Studio. Native runtime; no Python host.
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <VideoToolbox/VideoToolbox.h>
#import <CoreText/CoreText.h>
#import <IOSurface/IOSurface.h>
#import <obs-module.h>
#import "dlss-nr.h"
#import "DLSSControls.h"
#import "MacDisplayCapture.h"
#import "dlss5-nr-filter.h"
#import <obs-frontend-api.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-dlss-nr","en-US")
MODULE_EXPORT const char *obs_module_description(void) { return "DLSS 5 NR: obs-dlss5-nr native model pipeline, HDMI / Mac display capture and frame generation"; }

static NSString *resource_path(NSString *key) {
    NSString *binary=@(obs_get_module_binary_path(obs_current_module()));
    NSString *root=[[[binary stringByDeletingLastPathComponent] stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"Resources"];
    NSString *file=[@{@"model71":@"NeuralRendering.dlssmodel",@"model31":@"NeuralRendering-Small.dlssmodel",@"srPath":@"rt4ksr-x3.safetensors",@"frameGenerationPath":@"framegen.safetensors",@"rifeModelPath":@"rife-v4.25.rmw"} objectForKey:key];
    return [root stringByAppendingPathComponent:file];
}
static NSString *configuration_json(obs_data_t *settings) {
    NSMutableDictionary *d=[NSMutableDictionary dictionary];
    d[@"rifeModelPath"]=resource_path(@"rifeModelPath");
    for(NSString *key in @[@"mode",@"intensityMode",@"model71",@"model31",@"srPath",@"frameGenerationPath",@"frameGenerationBackend"])d[key]=@(obs_data_get_string(settings,key.UTF8String));
    for(NSString *key in @[@"enabled",@"upscaling",@"automaticMask",@"captureGuided",@"temporalNR",@"uiCorrection"])d[key]=obs_data_get_bool(settings,key.UTF8String) ? @YES : @NO;
    for(NSString *key in @[@"style",@"noiseFrame",@"frameGenerationFactor",@"frameGenerationHeight",@"nativeProcessingHeight"])d[key]=@(obs_data_get_int(settings,key.UTF8String));
    for(NSString *key in @[@"diffuseWhite",@"hdrStrength",@"colorStrength",@"uiX",@"uiY",@"uiWidth",@"uiHeight",@"motionScaleX",@"motionScaleY",@"intensity",@"layerIntensity",@"tone",@"structure",@"skinStructure",@"maskStructure"])d[key]=@(obs_data_get_double(settings,key.UTF8String));
    return [[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:d options:0 error:nil] encoding:NSUTF8StringEncoding];
}
@class NeuralCapture;
static NSDictionary *runtime_status(NeuralCapture *capture);
static void capture_instance(void *data,calldata_t *params);

@interface NeuralCapture : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(strong) AVCaptureSession *session;
@property(atomic, strong) MacDisplayCapture *displayCapture;
@property(atomic, copy) NSString *captureInput;
@property uint32_t requestedDisplay;
@property(strong) dispatch_queue_t captureQueue;
@property(strong) dispatch_queue_t controlQueue;
@property(strong) dispatch_queue_t diagnosticsQueue;
@property BOOL diagnosticsBusy;
@property BOOL authorizationPending;
@property BOOL statusEnabled;
@property gs_texture_t *statusTexture;
@property(copy) NSString *statusMessage;
@property void *engine;
@property uint64_t sequence;
@property gs_texture_t *texture;
@property void *frameLease;
@property(copy) NSString *deviceID;
@property(copy) NSString *modelPath;
@property(copy) NSString *appearancePath;
@property(copy) NSString *anePath;
@property(atomic, copy) NSDictionary *captureColor;
@property(copy) NSDictionary *effectiveColor;
@property VTPixelTransferSessionRef colorTransfer;
@property CVPixelBufferPoolRef colorPool;
@property(copy) NSString *captureError;
@property uint64_t renderedSequence,renderedUnique,captureStarts,captureDrops,renderedBase,renderedGenerated,renderedGuided,renderedRepeats;
@property double renderStart,lastStatusTime;
@property obs_source_t *owner;
- (void)configure:(obs_data_t *)settings;
- (void)shutdown;
- (void)consumeSample:(CMSampleBufferRef)sample clock:(CMClockRef)clock;
@end

@implementation NeuralCapture
- (instancetype)init {
    if ((self=[super init])) {
        _captureQueue=dispatch_queue_create("obs.neural.capture",DISPATCH_QUEUE_SERIAL);
        _controlQueue=dispatch_queue_create("obs.neural.control",DISPATCH_QUEUE_SERIAL);
        _diagnosticsQueue=dispatch_queue_create("obs.neural.diagnostics",dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL,QOS_CLASS_UTILITY,0));
        _captureError=@"";
    }
    return self;
}
- (void)releaseColorTransfer {
    if (_colorTransfer) CFRelease(_colorTransfer);
    if (_colorPool) CFRelease(_colorPool);
    _colorTransfer = NULL;
    _colorPool = NULL;
    self.captureColor = nil;
    self.effectiveColor = nil;
}
- (BOOL)prepareColorTransfer {
    OSStatus result = VTPixelTransferSessionCreate(NULL, &_colorTransfer);
    if (result == noErr)
        result = VTSessionSetProperty(_colorTransfer, kVTPixelTransferPropertyKey_DestinationColorPrimaries,
                                      kCVImageBufferColorPrimaries_ITU_R_709_2);
    if (result == noErr)
        result = VTSessionSetProperty(_colorTransfer, kVTPixelTransferPropertyKey_DestinationTransferFunction,
                                      kCVImageBufferTransferFunction_sRGB);
    if (result == noErr) {
        NSDictionary *attributes = @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
            (id)kCVPixelBufferWidthKey: @1920, (id)kCVPixelBufferHeightKey: @1080,
            (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
            (id)kCVPixelBufferMetalCompatibilityKey: @YES
        };
        result = CVPixelBufferPoolCreate(NULL, NULL, (__bridge CFDictionaryRef)attributes, &_colorPool);
    }
    if (result == noErr) return YES;
    [self releaseColorTransfer];
    self.captureError = [NSString stringWithFormat:@"Cannot initialize SDR color conversion (%d)", (int)result];
    return NO;
}
- (void)configure:(obs_data_t *)settings {
    NSString *deviceID=@(obs_data_get_string(settings,"device"));
    NSString *json=configuration_json(settings);
    NSString *input=[@(obs_data_get_string(settings,"captureInput")) isEqual:@"display"] ? @"display":@"hdmi";
    uint32_t display=(uint32_t)obs_data_get_int(settings,"captureDisplay");
    bool sameDevice=[_deviceID isEqual:deviceID]&&[_captureInput isEqual:input]&&_requestedDisplay==display;
    _deviceID=deviceID;_captureInput=input;_requestedDisplay=display;
    dispatch_sync(_controlQueue, ^{
        self.captureError=@"";
        @synchronized(self) {
            if(self.engine) {if(!nr_update(self.engine,json.UTF8String))self.captureError=@"Invalid DLSS NR settings";}
            else self.engine=nr_create(json.UTF8String);
            nr_set_status_overlay(self.engine,obs_data_get_bool(settings,"onScreenStatus"));
        }
        if(!self.engine) {self.captureError=@"Could not create DLSS NR runtime";return;}
        BOOL show=obs_data_get_bool(settings,"onScreenStatus");
        NSString *message=obs_data_get_bool(settings,"enabled") ? @"DLSS ON":@"DLSS OFF";
        if(show && (!self.statusTexture || ![self.statusMessage isEqual:message])) {
            const size_t width=260,height=60;NSMutableData *pixels=[NSMutableData dataWithLength:width*height*4];
            CGColorSpaceRef space=CGColorSpaceCreateDeviceRGB();
            CGContextRef context=CGBitmapContextCreate(pixels.mutableBytes,width,height,8,width*4,space,(CGBitmapInfo)kCGImageAlphaPremultipliedFirst|kCGBitmapByteOrder32Little);
            CGColorSpaceRelease(space);
            if(context){
                CGContextSetRGBFillColor(context,0,0,0,1);CGContextFillRect(context,CGRectMake(0,0,width,height));
                CTFontRef font=CTFontCreateWithName(CFSTR("Helvetica-Bold"),30,NULL);
                CGFloat rgba[]={.69,1,.28,1};CGColorSpaceRef colorSpace=CGColorSpaceCreateDeviceRGB();CGColorRef color=CGColorCreate(colorSpace,rgba);CGColorSpaceRelease(colorSpace);
                NSDictionary *attrs=@{(__bridge NSString *)kCTFontAttributeName:(__bridge id)font,(__bridge NSString *)kCTForegroundColorAttributeName:(__bridge id)color};
                NSAttributedString *text=[[NSAttributedString alloc] initWithString:message attributes:attrs];CTLineRef line=CTLineCreateWithAttributedString((__bridge CFAttributedStringRef)text);
                CGContextSetTextPosition(context,16,19);CTLineDraw(line,context);
                CFRelease(line);CFRelease(font);CGColorRelease(color);CGContextRelease(context);
                const uint8_t *bytes=pixels.bytes;obs_enter_graphics();
                if(self.statusTexture)gs_texture_set_image(self.statusTexture,bytes,width*4,false);
                else self.statusTexture=gs_texture_create(width,height,GS_BGRA,1,&bytes,GS_DYNAMIC);
                obs_leave_graphics();self.statusMessage=message;
            }
        }
        self.statusEnabled=show;
        BOOL displayMode=[input isEqual:@"display"];
        if(sameDevice&&((displayMode&&(self.displayCapture.running||self.displayCapture.starting))||(!displayMode&&self.session.isRunning)))return;
        [self.displayCapture stop];self.displayCapture=nil;
        [self.session stopRunning];self.session=nil;
        dispatch_sync(self.captureQueue,^{[self releaseColorTransfer];});
        nr_reset_input(self.engine);
        if(displayMode) {
            if(![self prepareColorTransfer])return;
            __weak NeuralCapture *weak=self;
            self.displayCapture=[[MacDisplayCapture alloc] initWithControlQueue:self.controlQueue captureQueue:self.captureQueue
                frame:^(CMSampleBufferRef sample){[weak consumeSample:sample clock:NULL];}
                error:^(NSString *message){weak.captureError=message;}];
            self.captureStarts++;
            [self.displayCapture startDisplay:display];
            return;
        }
        AVAuthorizationStatus authorization=[AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
        if(authorization==AVAuthorizationStatusNotDetermined) {
            self.captureError=@"Allow camera access in the macOS dialog to use HDMI capture.";
            if(!self.authorizationPending) {
                self.authorizationPending=YES;obs_source_t *lease=obs_source_get_ref(self.owner);
                dispatch_async(dispatch_get_main_queue(),^{[AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted){
                    dispatch_async(dispatch_get_main_queue(),^{
                        self.authorizationPending=NO;
                        if(granted){obs_data_t *current=obs_source_get_settings(lease);[self configure:current];obs_data_release(current);}
                        else self.captureError=@"Camera access denied. Enable OBS in System Settings → Privacy & Security → Camera.";
                        obs_source_release(lease);
                    });
                }];});
            }
            return;
        }
        if(authorization!=AVAuthorizationStatusAuthorized){self.captureError=@"Camera access is disabled in macOS Privacy & Security → Camera.";return;}
        if(!deviceID.length) return;
        AVCaptureDevice *device=[AVCaptureDevice deviceWithUniqueID:deviceID];
        if(!device) {
            AVCaptureDeviceDiscoverySession *discovery=[AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeExternal] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
            for(AVCaptureDevice *candidate in discovery.devices)if([candidate.uniqueID isEqual:deviceID]){device=candidate;break;}
        }
        if (!device) { self.captureError=@"Selected capture device is unavailable"; return; }
        NSError *error=nil;
        AVCaptureDeviceInput *input=[AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
        if (!input) { self.captureError=error.localizedDescription; return; }
        AVCaptureSession *session=[AVCaptureSession new];
        [session beginConfiguration];
        if (![session canAddInput:input]) { self.captureError=@"Cannot add capture input"; [session commitConfiguration]; return; }
        [session addInput:input];
        if ([session canSetSessionPreset:AVCaptureSessionPreset1920x1080]) session.sessionPreset=AVCaptureSessionPreset1920x1080;
        AVCaptureDeviceFormat *chosen=nil;
        CMTime duration=kCMTimeInvalid;
        const CMTime targetDuration=CMTimeMake(1,60);
        for (AVCaptureDeviceFormat *format in device.formats) {
            CMVideoDimensions d=CMVideoFormatDescriptionGetDimensions(format.formatDescription);
            if(d.width!=1920||d.height!=1080) continue;
            for(AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
                if(CMTimeCompare(range.maxFrameDuration,targetDuration)<0) continue;
                CMTime candidate=CMTimeMaximum(range.minFrameDuration,targetDuration);
                if(!chosen || CMTimeCompare(candidate,duration)<0) {
                    chosen=format; duration=candidate;
                }
            }
        }
        if(!chosen) { self.captureError=@"Device has no supported 1080p capture mode"; [session commitConfiguration]; return; }
        AVCaptureVideoDataOutput *output=[AVCaptureVideoDataOutput new];
        output.alwaysDiscardsLateVideoFrames=YES;
        [output setSampleBufferDelegate:self queue:self.captureQueue];
        if(![session canAddOutput:output]) { self.captureError=@"Cannot add video output"; [session commitConfiguration]; return; }
        [session addOutput:output];
        // On macOS the output connection has its own cadence. Device format alone
        // left this USB card delivering 1080p20 in the real PS5 capture test.
        AVCaptureConnection *connection=[output connectionWithMediaType:AVMediaTypeVideo];
        if(connection.supportsVideoMinFrameDuration) connection.videoMinFrameDuration=duration;
        if(connection.supportsVideoMaxFrameDuration) connection.videoMaxFrameDuration=duration;
        if(![device lockForConfiguration:&error]) { self.captureError=error.localizedDescription; [session commitConfiguration]; return; }
        device.activeFormat=chosen;
        device.activeVideoMinFrameDuration=duration; device.activeVideoMaxFrameDuration=duration;
        [device unlockForConfiguration];
        NSArray<NSNumber *> *formats = output.availableVideoCVPixelFormatTypes;
        NSNumber *format = @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange);
        if (![formats containsObject:format]) format = @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
        if (![formats containsObject:format]) {
            self.captureError = @"Capture device offers no 1080p NV12 output";
            [session commitConfiguration];
            return;
        }
        if (![self prepareColorTransfer]) { [session commitConfiguration]; return; }
        output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey: format,
                                 (id)kCVPixelBufferMetalCompatibilityKey: @YES,
                                 (id)kCVPixelBufferWidthKey: @1920, (id)kCVPixelBufferHeightKey: @1080};

        [session commitConfiguration];
        self.session=session; self.captureStarts++; [session startRunning];
        if(!session.isRunning)self.captureError=@"Capture session failed to start. Check camera permission and the device connection.";
        // Keep the configured device format fixed while the stream is running.
        // Real-device relaunch tests exposed an intermittent capture stall when
        // activeFormat was written again immediately after startRunning.
    });
}
- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sample fromConnection:(AVCaptureConnection *)connection {
    (void)output; (void)connection;
    [self consumeSample:sample clock:self.session.synchronizationClock];
}
- (void)consumeSample:(CMSampleBufferRef)sample clock:(CMClockRef)clock {
    CVPixelBufferRef pixels=CMSampleBufferGetImageBuffer(sample);
    if(!pixels) return;
    OSType format = CVPixelBufferGetPixelFormatType(pixels);
    if(CVPixelBufferGetWidth(pixels)!=1920 || CVPixelBufferGetHeight(pixels)!=1080 ||
       (format != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange &&
        format != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)) {
        self.captureError=@"Capture negotiated outside the required 1080p NV12 contract";
        return;
    }
    CMTime pts=CMSampleBufferGetPresentationTimeStamp(sample);
    if(clock && clock!=CMClockGetHostTimeClock()) pts=CMSyncConvertTime(pts,clock,CMClockGetHostTimeClock());
    if(!CMTIME_IS_NUMERIC(pts)) return;
    CFStringRef keys[] = {kCVImageBufferYCbCrMatrixKey, kCVImageBufferColorPrimariesKey,
                          kCVImageBufferTransferFunctionKey};
    NSString *names[] = {@"matrix", @"primaries", @"transfer"};
    NSString *defaults[] = {(__bridge NSString *)kCVImageBufferYCbCrMatrix_ITU_R_709_2,
                            (__bridge NSString *)kCVImageBufferColorPrimaries_ITU_R_709_2,
                            (__bridge NSString *)kCVImageBufferTransferFunction_ITU_R_709_2};
    CMFormatDescriptionRef description = CMSampleBufferGetFormatDescription(sample);
    NSMutableDictionary *declared = [NSMutableDictionary dictionary];
    NSMutableDictionary *effective = [NSMutableDictionary dictionary];
    BOOL needsAttachment[3] = {NO, NO, NO};
    for (NSUInteger i = 0; i < 3; i++) {
        id tag = CFBridgingRelease(CVBufferCopyAttachment(pixels, keys[i], NULL));
        needsAttachment[i] = tag == nil;
        if (!tag && description)
            tag = (__bridge id)CMFormatDescriptionGetExtension(description, keys[i]);
        declared[names[i]] = tag ?: @"untagged";
        effective[names[i]] = tag ?: self.effectiveColor[names[i]] ?: defaults[i];
    }
    self.effectiveColor = effective;
    NSDictionary *color = @{
        @"range": format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ? @"video" : @"full",
        @"declared": declared,
        @"effective": effective
    };
    if (![self.captureColor isEqualToDictionary:color]) {
        blog(LOG_INFO, "[DLSS NR] 1080p NV12 input %s", color.description.UTF8String);
        self.captureColor = color;
    }
    CVPixelBufferRef rgb = NULL;
    NSDictionary *limit = @{(id)kCVPixelBufferPoolAllocationThresholdKey: @6};
    OSStatus result = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(NULL, _colorPool,
        (__bridge CFDictionaryRef)limit, &rgb);
    if (result == kCVReturnWouldExceedAllocationThreshold) { self.captureDrops++; return; }
    if (result == noErr) {
        for (NSUInteger i = 0; i < 3; i++)
            if (needsAttachment[i])
                CVBufferSetAttachment(pixels, keys[i], (__bridge CFTypeRef)effective[names[i]], kCVAttachmentMode_ShouldPropagate);
        result = VTPixelTransferSessionTransferImage(_colorTransfer, pixels, rgb);
        for (NSUInteger i = 0; i < 3; i++)
            if (needsAttachment[i]) CVBufferRemoveAttachment(pixels, keys[i]);
    }
    if (result == noErr) {
        nr_submit(_engine, (void *)rgb, ++_sequence, CMTimeGetSeconds(pts));
    } else {
        self.captureError = [NSString stringWithFormat:@"NV12 color conversion failed (%d)", (int)result];
    }
    if (rgb) CFRelease(rgb);
}
- (void)captureOutput:(AVCaptureOutput *)output didDropSampleBuffer:(CMSampleBufferRef)sample fromConnection:(AVCaptureConnection *)connection {
    (void)output;(void)sample;(void)connection;self.captureDrops++;
}
- (void)shutdown {
    dispatch_sync(_controlQueue, ^{
        [self.displayCapture stop];self.displayCapture=nil;
        [self.session stopRunning]; self.session=nil;
        dispatch_sync(self.captureQueue, ^{});
        [self releaseColorTransfer];
        @synchronized(self) { nr_destroy(self.engine); self.engine=NULL; }
    });
    if(_texture) { obs_enter_graphics(); gs_texture_destroy(_texture); _texture=NULL; obs_leave_graphics(); }
    nr_frame_release(_frameLease); _frameLease=NULL;
    if(_statusTexture){obs_enter_graphics();gs_texture_destroy(_statusTexture);_statusTexture=NULL;obs_leave_graphics();}
}
@end

static NSDictionary *runtime_status(NeuralCapture *capture) {
    NSMutableData *buffer=[NSMutableData dataWithLength:262144];
    void *handle=NULL;
    @synchronized(capture) {handle=nr_handle_retain(capture.engine);}
    if(handle) {nr_status(handle,buffer.mutableBytes,(int32_t)buffer.length);nr_handle_release(handle);}

    NSString *text=[NSString stringWithUTF8String:buffer.bytes];
    NSMutableDictionary *status=[[NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:NSJSONReadingMutableContainers error:nil] mutableCopy] ?: [NSMutableDictionary dictionary];
    status[@"sampledAtHostSeconds"]=@(NSProcessInfo.processInfo.systemUptime);
    status[@"obsRenderedUnique"]=@(capture.renderedUnique);
    status[@"obsSelectedGuided"]=@(capture.renderedGuided);
    status[@"obsSelectedBase"]=@(capture.renderedBase);status[@"obsSelectedGenerated"]=@(capture.renderedGenerated);status[@"obsRepeatedDraws"]=@(capture.renderedRepeats);
    status[@"statusOverlayEnabled"]=@(capture.statusEnabled);status[@"statusOverlayAvailable"]=@(capture.statusTexture!=NULL);
    status[@"applicationBundleID"]=NSBundle.mainBundle.bundleIdentifier ?: @"";
    status[@"captureInput"]=capture.captureInput ?: @"hdmi";
    status[@"captureDisplayID"]=@(capture.displayCapture.displayID);
    status[@"screenCaptureAuthorized"]=@(CGPreflightScreenCaptureAccess());
    status[@"captureRunning"]=@(capture.session.isRunning||capture.displayCapture.running);status[@"captureAuthorization"]=@([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo]);status[@"engineAvailable"]=@(capture.engine!=NULL);
    status[@"captureStarts"]=@(capture.captureStarts);status[@"captureDrops"]=@(capture.captureDrops);
    status[@"captureColor"]=capture.captureColor ?: @{};
    status[@"captureError"]=capture.captureError ?: @"";
    return status;
}
static const char *source_name(void *unused) { (void)unused; return "DLSS 5 NR · HDMI / Mac display"; }
static void source_defaults(obs_data_t *settings) {
    obs_data_set_default_string(settings,"captureInput","hdmi");
    obs_data_set_default_int(settings,"captureDisplay",0);
    obs_data_set_default_string(settings,"mode","31then71-480");
    obs_data_set_default_string(settings,"preset","standard");
    obs_data_set_default_bool(settings,"enabled",true);
    obs_data_set_default_bool(settings,"upscaling",true);
    obs_data_set_default_int(settings,"style",0);
    obs_data_set_default_string(settings,"intensityMode","final");
    obs_data_set_default_double(settings,"intensity",1);
    obs_data_set_default_double(settings,"layerIntensity",1);
    obs_data_set_default_bool(settings,"temporalNR",false);
    obs_data_set_default_double(settings,"motionScaleX",1);
    obs_data_set_default_double(settings,"motionScaleY",1);
    obs_data_set_default_double(settings,"tone",1);
    obs_data_set_default_double(settings,"structure",1);
    obs_data_set_default_double(settings,"skinStructure",-1);
    obs_data_set_default_double(settings,"maskStructure",-1);
    obs_data_set_default_bool(settings,"automaticMask",false);
    obs_data_set_default_int(settings,"noiseFrame",0);
    obs_data_set_default_bool(settings,"onScreenStatus",false);
    obs_data_set_default_bool(settings,"captureGuided",false);
    obs_data_set_default_string(settings,"frameGenerationBackend","mlx");
    obs_data_set_default_int(settings,"frameGenerationFactor",1);
    obs_data_set_default_int(settings,"frameGenerationHeight",360);
    obs_data_set_default_int(settings,"nativeProcessingHeight",1080);
    obs_data_set_default_double(settings,"diffuseWhite",80);
    obs_data_set_default_double(settings,"uiX",4.9);
    obs_data_set_default_double(settings,"uiY",79);
    obs_data_set_default_double(settings,"uiWidth",14.7);
    obs_data_set_default_double(settings,"uiHeight",16.2);
    obs_data_set_default_double(settings,"hdrStrength",1);
    obs_data_set_default_double(settings,"colorStrength",1);
    for(NSString *key in @[@"model71",@"model31",@"srPath",@"frameGenerationPath"])obs_data_set_default_string(settings,key.UTF8String,resource_path(key).UTF8String);
    AVCaptureDeviceDiscoverySession *discovery=[AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeExternal] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
    for(AVCaptureDevice *device in discovery.devices)if([device.uniqueID hasSuffix:@"345f2130"]) {obs_data_set_default_string(settings,"device",device.uniqueID.UTF8String);break;}
}
static void *source_create(obs_data_t *settings, obs_source_t *source) {
    NeuralCapture *capture=[NeuralCapture new]; capture.owner=source; [capture configure:settings];
    proc_handler_add(obs_source_get_proc_handler(source),"void dlss_nr_capture(ptr capture)",capture_instance,(__bridge void *)capture);
    return (__bridge_retained void *)capture;
}
static void capture_instance(void *data,calldata_t *params) {NeuralCapture *capture=(__bridge NeuralCapture *)data;calldata_set_ptr(params,"capture",(__bridge_retained void *)capture);}
static void source_destroy(void *data) {
    NeuralCapture *capture=CFBridgingRelease(data); [capture shutdown];
}
static void source_update(void *data,obs_data_t *settings) {
    [(__bridge NeuralCapture *)data configure:settings];
}
static void schedule_diagnostics(NeuralCapture *capture,double now) {
    if(now-capture.lastStatusTime>=5) {
        capture.lastStatusTime=now;
        NSString *dir=NSProcessInfo.processInfo.environment[@"DLSS_NR_DIAGNOSTICS_DIR"];
        // Frame rendering must not serialize history or perform file I/O. The
        // retained runtime lease prevents teardown racing a diagnostic job.
        if(dir.length && !capture.diagnosticsBusy) {
            capture.diagnosticsBusy=YES;
            dispatch_async(capture.diagnosticsQueue, ^{
                @autoreleasepool {
                    NSDictionary *status=runtime_status(capture);
                    blog(LOG_INFO,"DLSS 5 NR: %s %.2f ms, %.2f processed FPS, %llu completed, %llu failures",[status[@"mode"] UTF8String] ?: "stopped",[status[@"processingMedianMS"] doubleValue],[status[@"processedFPS"] doubleValue],[status[@"completed"] unsignedLongLongValue],[status[@"failed"] unsignedLongLongValue]);
                    if(status.count && [NSJSONSerialization isValidJSONObject:status]) {
                        [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
                        NSData *json=[NSJSONSerialization dataWithJSONObject:status options:NSJSONWritingPrettyPrinted error:nil];
                        [json writeToFile:[dir stringByAppendingPathComponent:@"runtime.json"] atomically:YES];
                    }
                    capture.diagnosticsBusy=NO;
                }
            });
        }
    }
}
static void source_render(void *data,gs_effect_t *unused) {
    (void)unused;
    NeuralCapture *capture=(__bridge NeuralCapture *)data;
    schedule_diagnostics(capture,NSProcessInfo.processInfo.systemUptime);
    void *frame=NULL;
    @synchronized(capture) { frame=nr_frame_acquire(capture.engine); }
    if(!frame) return;
    CVPixelBufferRef pixels=nr_frame_pixels(frame);
    IOSurfaceRef surface=CVPixelBufferGetIOSurface(pixels);
    // Fresh immutable output surfaces are not returned to a pixel-buffer pool.
    // Backend texture ownership therefore cannot race pooled buffer rewriting.
    if(!capture.texture) capture.texture=gs_texture_create_from_iosurface(surface);
    else if(frame!=capture.frameLease && !gs_texture_rebind_iosurface(capture.texture,surface)) {
        nr_frame_release(frame); return;
    }
    nr_frame_release(capture.frameLease); capture.frameLease=frame;
    if(!capture.texture) return;
    gs_effect_t *effect=obs_get_base_effect(gs_get_device_type()==GS_DEVICE_OPENGL ? OBS_EFFECT_DEFAULT_RECT : OBS_EFFECT_DEFAULT);
    bool previous=gs_framebuffer_srgb_enabled();
    bool linear=gs_get_linear_srgb(); gs_enable_framebuffer_srgb(linear);
    gs_eparam_t *param=gs_effect_get_param_by_name(effect,"image");
    if(linear) gs_effect_set_texture_srgb(param,capture.texture); else gs_effect_set_texture(param,capture.texture);
    while(gs_effect_loop(effect,"Draw")) gs_draw_sprite(capture.texture,0,1920,1080);
    if(capture.statusEnabled && capture.statusTexture){gs_matrix_push();gs_matrix_translate3f(24,24,0);gs_effect_set_texture(param,capture.statusTexture);while(gs_effect_loop(effect,"Draw"))gs_draw_sprite(capture.statusTexture,0,260,60);gs_matrix_pop();}
    uint64_t seq=nr_frame_sequence(frame);
    if(seq!=capture.renderedSequence) {capture.renderedSequence=seq;capture.renderedUnique++;if(nr_frame_kind(frame)){capture.renderedGenerated++;if(nr_frame_kind(frame)==2)capture.renderedGuided++;}else capture.renderedBase++;}
    else capture.renderedRepeats++;
    if(!capture.renderStart)capture.renderStart=NSProcessInfo.processInfo.systemUptime;
    gs_enable_framebuffer_srgb(previous);
}
static uint32_t source_width(void *data) { (void)data; return 1920; }
static uint32_t source_height(void *data) { (void)data; return 1080; }
static bool preview_clicked(obs_properties_t *p,obs_property_t *property,void *data) {
    (void)p; (void)property;
    NeuralCapture *capture=(__bridge NeuralCapture *)data;
    @synchronized(capture) { nr_show_preview(capture.engine); }
    return false;
}
static bool preset_changed(obs_properties_t *props,obs_property_t *property,obs_data_t *settings) {
    (void)property;
    bool automatic=obs_data_get_bool(settings,"automaticMask");
    obs_property_set_enabled(obs_properties_get(props,"skinStructure"),automatic);
    obs_property_set_enabled(obs_properties_get(props,"maskStructure"),automatic);
    return true;
}
static bool apply_preset(obs_properties_t *props,obs_property_t *property,void *data) {
    (void)props;(void)property;
    NeuralCapture *capture=(__bridge NeuralCapture *)data;
    if(!capture.owner)return false;
    obs_data_t *settings=obs_source_get_settings(capture.owner);
    const char *preset=obs_data_get_string(settings,"preset");
    if(strcmp(preset,"custom")) {
        obs_data_set_int(settings,"style",!strcmp(preset,"natural") ? 1 : !strcmp(preset,"cinematic") ? 2 : 0);
        obs_data_set_double(settings,"tone",!strcmp(preset,"neutral") ? 0 : 1);
        obs_data_set_double(settings,"structure",!strcmp(preset,"neutral") ? 0 : 1);
        obs_source_update(capture.owner,settings);
        obs_source_update_properties(capture.owner);
    }
    obs_data_release(settings);return true;
}
static bool refresh_status(obs_properties_t *props,obs_property_t *property,void *data) {
    (void)property;
    NeuralCapture *capture=(__bridge NeuralCapture *)data;
    NSDictionary *s=runtime_status(capture);
    NSString *message=[NSString stringWithFormat:@"%@ · %.1f processed FPS · %.1f ms median · %@ completed · %@ errors\n%@%@",s[@"mode"] ?: @"Initializing",[s[@"processedFPS"] doubleValue],[s[@"processingMedianMS"] doubleValue],s[@"completed"] ?: @0,s[@"failed"] ?: @0,s[@"backend"] ?: @"MLX/Metal",capture.captureError.length ? [@"\n" stringByAppendingString:capture.captureError] : s[@"error"] ?: @""];
    message=[message stringByAppendingFormat:@"\nMFG %@× ·%@ generated ·%@ FG errors ·OBS selected %@ base /%@ generated ·%@ repeated draws",s[@"mfgFactor"] ?: @1,s[@"generatedFrames"] ?: @0,s[@"mfgFailures"] ?: @0,s[@"obsSelectedBase"] ?: @0,s[@"obsSelectedGenerated"] ?: @0,s[@"obsRepeatedDraws"] ?: @0];
    if([s[@"captureGuided"] boolValue])message=[message stringByAppendingFormat:@"\nCapture-guided: %@ reconstructed ·%@ capture-copy failures",s[@"guidedFrames"] ?: @0,s[@"captureGuides"][@"failed"] ?: @0];
    if([s[@"mfgBackend"] isEqual:@"apple"])message=[message stringByAppendingFormat:@"\nApple interpolation: %@ ·%@ skipped busy pairs ·%@",s[@"nativeFGState"] ?: @"inactive",s[@"nativeFGSkippedPairs"] ?: @0,s[@"nativeFGError"] ?: @""];
    NSDictionary *native=s[@"nativePreview"];
    if([native[@"active"] boolValue])message=[message stringByAppendingFormat:@"\nNative preview: %.1f base +%.1f generated shown FPS ·%.1f display Hz",[native[@"baseShownFPS"] doubleValue],[native[@"generatedShownFPS"] doubleValue],[native[@"displayHz"] doubleValue]];
    obs_property_set_description(obs_properties_get(props,"status"),message.UTF8String);
    return true;
}
static bool intensity_mode_changed(obs_properties_t *props,obs_property_t *property,obs_data_t *settings) {
    (void)property;
    bool layer=!strcmp(obs_data_get_string(settings,"intensityMode"),"layer");
    obs_property_set_enabled(obs_properties_get(props,"intensity"),!layer);
    obs_property_set_enabled(obs_properties_get(props,"layerIntensity"),layer);
    return true;
}
static bool display_permission_clicked(obs_properties_t *props,obs_property_t *property,void *data) {
    (void)props;(void)property;
    NeuralCapture *capture=(__bridge NeuralCapture *)data;
    if(!capture.owner)return false;
    obs_source_t *lease=obs_source_get_ref(capture.owner);
    dispatch_async(dispatch_get_main_queue(),^{
        CGRequestScreenCaptureAccess();
        obs_data_t *settings=obs_source_get_settings(lease);obs_source_update(lease,settings);
        obs_data_release(settings);obs_source_release(lease);
    });
    return false;
}
static obs_properties_t *source_properties(void *data) {
    obs_properties_t *p=obs_properties_create();
    obs_property_t *input=obs_properties_add_list(p,"captureInput","Input source",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(input,"PS5 / external HDMI","hdmi");
    obs_property_list_add_string(input,"Mac display · ScreenCaptureKit","display");
    obs_property_t *display=obs_properties_add_list(p,"captureDisplay","Mac display ·1080p SDR",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(display,"Automatic ·built-in / primary",0);
    for(NSScreen *screen in NSScreen.screens) {
        NSNumber *number=screen.deviceDescription[@"NSScreenNumber"];
        obs_property_list_add_int(display,screen.localizedName.UTF8String,number.unsignedIntValue);
    }
    obs_properties_add_button2(p,"displayPermission","Screen recording permission / reconnect display",display_permission_clicked,data);
    obs_property_t *devices=obs_properties_add_list(p,"device","HDMI capture device ·1080p",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(devices,"Select capture device","");
    AVCaptureDeviceDiscoverySession *discovery=[AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeExternal] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
    for(AVCaptureDevice *device in discovery.devices)obs_property_list_add_string(devices,device.localizedName.UTF8String,device.uniqueID.UTF8String);
    obs_properties_t *nr=obs_properties_create();
    obs_properties_add_bool(nr,"enabled","Enable DLSS Neural Rendering");
    obs_property_t *m=obs_properties_add_list(nr,"mode","DLSS mode",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(m,"DLSS Fidelity ·71 →71", "71x2-480");
    obs_property_list_add_string(m,"DLSS Balanced ·31 →71", "31then71-480");
    obs_property_list_add_string(m,"DLSS Performance ·31 single", "31-480");
    obs_property_list_add_string(m,"DLSS 71 Single ·71", "71-480");
    obs_property_list_add_string(m,"DLSS 31 Double ·31 →31", "31x2-480");
    obs_properties_add_bool(nr,"upscaling","Enable learned upscaling (off = spatial resize)");
    obs_property_t *preset=obs_properties_add_list(nr,"preset","NR Preset",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(preset,"Default / Standard","standard");obs_property_list_add_string(preset,"Natural","natural");obs_property_list_add_string(preset,"Cinematic","cinematic");obs_property_list_add_string(preset,"Neutral","neutral");obs_property_list_add_string(preset,"Custom","custom");
    obs_properties_add_button2(nr,"applyPreset","Apply selected NR preset",apply_preset,data);
    obs_properties_add_int(nr,"style","NR Style index",0,128,1);
    obs_property_t *intensityMode=obs_properties_add_list(nr,"intensityMode","Intensity Mode",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(intensityMode,"Final","final");
    obs_property_list_add_string(intensityMode,"Per-Layer","layer");
    obs_property_set_modified_callback(intensityMode,intensity_mode_changed);
    obs_properties_add_float_slider(nr,"intensity","Final Intensity",0,2,.01);
    obs_properties_add_float_slider(nr,"layerIntensity","Per-Layer Intensity",0,2,.01);
    obs_properties_add_float_slider(nr,"tone","Local Tone Strength",0,2,.01);
    obs_properties_add_float_slider(nr,"structure","Local Structure Strength",0,2,.01);
    obs_property_t *mask=obs_properties_add_bool(nr,"automaticMask","Automatic Mask conditioning");obs_property_set_modified_callback(mask,preset_changed);
    obs_properties_add_float_slider(nr,"skinStructure","Skin Structure Strength (-1 = inherit)",-1,2,.01);
    obs_properties_add_float_slider(nr,"maskStructure","Automatic Mask Structure (-1 = inherit)",-1,2,.01);
    obs_properties_add_int(nr,"noiseFrame","Noise seed / frame index (fixed)",0,65535,1);
    obs_properties_add_group(p,"nr","DLSS 5 NR · Neural Rendering",OBS_GROUP_NORMAL,nr);
    obs_properties_t *guides=obs_properties_create();
    obs_properties_add_bool(guides,"temporalNR","Screen-space temporal NR · experimental");
    obs_properties_add_float_slider(guides,"motionScaleX","Screen-space motion scale X",-2,2,.01);
    obs_properties_add_float_slider(guides,"motionScaleY","Screen-space motion scale Y",-2,2,.01);
    obs_properties_add_text(guides,"guide_info","1080p SDR source → native NR pipeline. The processing maximum controls model input sizing. Engine depth/motion are unavailable.",OBS_TEXT_INFO);
    obs_properties_add_bool(guides,"uiCorrection","Protect a screen-space HUD rectangle");
    obs_properties_add_float(guides,"uiX","HUD X (%)",0,100,.1);
    obs_properties_add_float(guides,"uiY","HUD Y (%)",0,100,.1);
    obs_properties_add_float(guides,"uiWidth","HUD width (%)",0,100,.1);
    obs_properties_add_float(guides,"uiHeight","HUD height (%)",0,100,.1);
    obs_properties_add_float(guides,"diffuseWhite","SDR diffuse white reference (nits)",80,400,1);
    obs_properties_add_float(guides,"hdrStrength","OKLab lightness correction",0,2,.01);
    obs_properties_add_float(guides,"colorStrength","OKLab chroma correction",0,2,.01);
    obs_properties_add_text(guides,"depthConvention","Screen-space motion uses captured images. No engine depth input is available.",OBS_TEXT_INFO);
    obs_properties_add_group(p,"guides","Color transfer and guide availability",OBS_GROUP_NORMAL,guides);
    obs_property_t *nrsize=obs_properties_add_list(p,"nativeProcessingHeight","Native NR processing resolution",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(nrsize,"Source",0);
    obs_property_list_add_int(nrsize,"480p",480);
    obs_property_list_add_int(nrsize,"720p",720);
    obs_property_list_add_int(nrsize,"1080p",1080);
    obs_property_list_add_int(nrsize,"1440p",1440);
    obs_property_list_add_int(nrsize,"2160p",2160);
    obs_properties_t *fg=obs_properties_create();
    obs_property_t *backend=obs_properties_add_list(fg,"frameGenerationBackend","Interpolation engine",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(backend,"Apple VideoToolbox ·native motion reconstruction","apple");
    obs_property_list_add_string(backend,"MLX ·experimental recovered video model","mlx");
    obs_property_list_add_string(backend,"RIFE Metal v4.25","rife");
    obs_property_t *factor=obs_properties_add_list(fg,"frameGenerationFactor","Multi-frame generation",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(factor,"Off ·base frames only",1);
    obs_property_list_add_int(factor,"2× ·one intermediate per pair",2);
    obs_property_list_add_int(factor,"3× ·two intermediates per pair",3);
    obs_property_list_add_int(factor,"4× ·three intermediates per pair",4);
    obs_property_list_add_int(factor,"5× ·four intermediates per pair",5);
    obs_property_list_add_int(factor,"6× maximum ·up to five intermediates per pair",6);
    obs_property_t *flow=obs_properties_add_list(fg,"frameGenerationHeight","MLX motion-estimation detail ·Apple uses full1080p",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(flow,"Fast ·360p motion estimate",360);
    obs_property_list_add_int(flow,"Detailed ·540p motion estimate",540);
    obs_property_list_add_int(flow,"High ·720p motion estimate",720);
    obs_properties_add_text(fg,"fgInfo","Learned interpolation after NR. Requires two completed base frames and adds playback delay. Factors describe frames per pair, not a guaranteed output rate. OBS output FPS still limits composition; the native preview can present up to120Hz.",OBS_TEXT_INFO);
    obs_properties_add_path(fg,"frameGenerationPath","Frame-generation weights",OBS_PATH_FILE,"SafeTensors (*.safetensors)",NULL);
    obs_properties_add_group(p,"fg","Multi-Frame Generation ·experimental",OBS_GROUP_NORMAL,fg);
    obs_properties_t *models=obs_properties_create();
    obs_properties_add_path(models,"model71","71 HT checkpoint",OBS_PATH_DIRECTORY,NULL,NULL);
    obs_properties_add_path(models,"model31","31 HT Small checkpoint",OBS_PATH_DIRECTORY,NULL,NULL);
    obs_properties_add_path(models,"srPath","Reconstruction weights",OBS_PATH_FILE,"SafeTensors (*.safetensors)",NULL);
    obs_properties_add_group(p,"models","Model configuration",OBS_GROUP_NORMAL,models);
    obs_properties_add_button2(p,"preview","Open native ProMotion preview",preview_clicked,data);
    obs_properties_add_text(p,"status","NR MLX/Metal FP16 + MPS ·NR ANE off ·MFG optional",OBS_TEXT_INFO);
    obs_properties_add_button2(p,"refresh","Refresh runtime measurements",refresh_status,data);
    if(data)refresh_status(p,NULL,data);
    for(NSString *key in @[@"upscaling",@"diffuseWhite",@"hdrStrength",@"colorStrength",@"motionScaleX",@"motionScaleY",@"noiseFrame",@"uiCorrection",@"uiX",@"uiY",@"uiWidth",@"uiHeight",@"srPath"]) {
        obs_property_t *property=obs_properties_get(p,key.UTF8String);
        if(property)obs_property_set_enabled(property,false);
    }
    return p;
}
static void open_nr_menu(void *unused) {
    (void)unused;
    obs_source_t *source=obs_get_source_by_name("DLSS 5 NR");
    if(!source) {
        obs_data_t *settings=obs_data_create();
        source=obs_source_create("mac_dlss5_nr","DLSS 5 NR",settings,NULL);obs_data_release(settings);
        obs_source_t *current=obs_frontend_get_current_scene();
        if(current) {obs_scene_t *scene=obs_scene_from_source(current);if(scene && source)obs_scene_add(scene,source);obs_source_release(current);}
    }
    if(source) {calldata_t params;calldata_init(&params);proc_handler_call(obs_source_get_proc_handler(source),"dlss_nr_capture",&params);NeuralCapture *capture=CFBridgingRelease(calldata_ptr(&params,"capture"));calldata_free(&params);dlss_controls_show(source,^NSDictionary *{return runtime_status(capture);});obs_source_release(source);}
}
static void frontend_event(enum obs_frontend_event event,void *unused) {
    (void)unused;
    if(event==OBS_FRONTEND_EVENT_SCRIPTING_SHUTDOWN || event==OBS_FRONTEND_EVENT_EXIT)dlss_controls_close();
}
bool obs_module_load(void) {
    if (nr_abi_version() != NR_ABI_VERSION) {
        blog(LOG_ERROR,"DLSS 5 NR: incompatible controller ABI");
        return false;
    }
    NSString *binary=@(obs_get_module_binary_path(obs_current_module()));
    NSString *metallib=[[[binary stringByDeletingLastPathComponent] stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"Resources/mlx.metallib"];
    if(![[NSFileManager defaultManager] fileExistsAtPath:metallib]) {
        blog(LOG_ERROR,"Neural Post: packaged MLX Metal library is missing"); return false;
    }
    nr_set_metallib(metallib.fileSystemRepresentation);
    struct obs_source_info info={0};
    info.id="mac_dlss5_nr"; info.type=OBS_SOURCE_TYPE_INPUT;
    info.output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_CUSTOM_DRAW;
    info.get_name=source_name; info.create=source_create; info.destroy=source_destroy;
    info.update=source_update; info.get_properties=source_properties;
    info.get_defaults=source_defaults;
    info.get_width=source_width; info.get_height=source_height; info.video_render=source_render;
    info.icon_type=OBS_ICON_TYPE_CAMERA;
    obs_register_source(&info);
    register_dlss5nr_filter();
    obs_frontend_add_tools_menu_item("DLSS 5",open_nr_menu,NULL);
    obs_frontend_add_event_callback(frontend_event,NULL);
    return true;
}
