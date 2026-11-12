#import "NativeInput.h"
#import "../nr/MacDisplayCapture.h"
#import <AVFoundation/AVFoundation.h>
#import <VideoToolbox/VideoToolbox.h>
@interface NativeInput () <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(atomic,copy,readwrite) NSString *error;
@property AVCaptureSession *session;
@property MacDisplayCapture *display;
@property dispatch_queue_t controlQueue,captureQueue;
@property(copy) void (^frame)(CVPixelBufferRef,double);
@property(atomic,copy) NSDictionary *captureColor;
@property(copy) NSDictionary *effectiveColor;
@property VTPixelTransferSessionRef colorTransfer;
@property CVPixelBufferPoolRef colorPool;
@property size_t width,height;
@property uint64_t frames,drops,captureStarts;
@property NSString *kind;
@end
@implementation NativeInput
- (instancetype)initWithFrame:(void (^)(CVPixelBufferRef,double))frame {
 if((self=[super init])){_frame=[frame copy];_error=@"";_controlQueue=dispatch_queue_create("nr.native.input.control",DISPATCH_QUEUE_SERIAL);_captureQueue=dispatch_queue_create("nr.native.input.frames",DISPATCH_QUEUE_SERIAL);}
 return self;
}
- (void)stop {
 dispatch_sync(self.controlQueue,^{
  [self.display stop];self.display=nil;[self.session stopRunning];self.session=nil;
  dispatch_sync(self.captureQueue,^{[self releaseColorTransfer];});
 });
}
- (void)startScreen:(CGDirectDisplayID)display {
 [self stop];dispatch_sync(self.controlQueue,^{
  self.error=@"";self.kind=@"screen";__weak NativeInput *weak=self;
  self.display=[[MacDisplayCapture alloc] initWithControlQueue:self.controlQueue captureQueue:self.captureQueue
    frame:^(CMSampleBufferRef sample){[weak consumeSample:sample clock:NULL];}
    error:^(NSString *message){weak.error=message;}];
  self.display.nativeAspect=YES;self.captureStarts++;[self.display startDisplay:display];
 });
}
- (void)startHDMI:(NSString *)deviceID {
 [self stop];dispatch_sync(self.controlQueue,^{
  self.kind=@"hdmi";self.error=@"";
  if([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo]!=AVAuthorizationStatusAuthorized){self.error=@"Camera permission required for HDMI capture";return;}
        AVCaptureDevice *device=[AVCaptureDevice deviceWithUniqueID:deviceID];
        if(!device) {
            AVCaptureDeviceDiscoverySession *discovery=[AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeExternal] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
            for(AVCaptureDevice *candidate in discovery.devices)if([candidate.uniqueID isEqual:deviceID]){device=candidate;break;}
        }
        if (!device) { self.error=@"Selected capture device is unavailable"; return; }
        NSError *error=nil;
        AVCaptureDeviceInput *input=[AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
        if (!input) { self.error=error.localizedDescription; return; }
        AVCaptureSession *session=[AVCaptureSession new];
        [session beginConfiguration];
        if (![session canAddInput:input]) { self.error=@"Cannot add capture input"; [session commitConfiguration]; return; }
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
        if(!chosen) { self.error=@"Device has no supported 1080p capture mode"; [session commitConfiguration]; return; }
        AVCaptureVideoDataOutput *output=[AVCaptureVideoDataOutput new];
        output.alwaysDiscardsLateVideoFrames=YES;
        [output setSampleBufferDelegate:self queue:self.captureQueue];
        if(![session canAddOutput:output]) { self.error=@"Cannot add video output"; [session commitConfiguration]; return; }
        [session addOutput:output];
        // On macOS the output connection has its own cadence. Device format alone
        // left this USB card delivering 1080p20 in the real PS5 capture test.
        AVCaptureConnection *connection=[output connectionWithMediaType:AVMediaTypeVideo];
        if(connection.supportsVideoMinFrameDuration) connection.videoMinFrameDuration=duration;
        if(connection.supportsVideoMaxFrameDuration) connection.videoMaxFrameDuration=duration;
        if(![device lockForConfiguration:&error]) { self.error=error.localizedDescription; [session commitConfiguration]; return; }
        device.activeFormat=chosen;
        device.activeVideoMinFrameDuration=duration; device.activeVideoMaxFrameDuration=duration;
        [device unlockForConfiguration];
        NSArray<NSNumber *> *formats = output.availableVideoCVPixelFormatTypes;
        NSNumber *format = @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange);
        if (![formats containsObject:format]) format = @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
        if (![formats containsObject:format]) {
            self.error = @"Capture device offers no 1080p NV12 output";
            [session commitConfiguration];
            return;
        }
        output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey: format,
                                 (id)kCVPixelBufferMetalCompatibilityKey: @YES,
                                 (id)kCVPixelBufferWidthKey: @1920, (id)kCVPixelBufferHeightKey: @1080};

        [session commitConfiguration];
        self.session=session; self.captureStarts++; [session startRunning];
        if(!session.isRunning)self.error=@"Capture session failed to start. Check camera permission and the device connection.";

 });
}
- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sample fromConnection:(AVCaptureConnection *)connection {
 (void)output;(void)connection;[self consumeSample:sample clock:self.session.synchronizationClock];
}
- (void)captureOutput:(AVCaptureOutput *)output didDropSampleBuffer:(CMSampleBufferRef)sample fromConnection:(AVCaptureConnection *)connection {
 (void)output;(void)sample;(void)connection;self.drops++;
}
- (NSDictionary *)status {
 return @{@"input":self.kind ?: @"stopped",@"running":@((BOOL)(self.display.running||self.session.isRunning)),
  @"error":self.error ?: @"",@"frames":@(self.frames),@"drops":@(self.drops),@"starts":@(self.captureStarts),
  @"width":@(self.width),@"height":@(self.height),@"color":self.captureColor ?: @{},@"displayID":@(self.display.displayID)};
}
- (void)releaseColorTransfer {
    if (_colorTransfer) CFRelease(_colorTransfer);
    if (_colorPool) CFRelease(_colorPool);
    _colorTransfer = NULL;
    _colorPool = NULL;
    self.captureColor = nil;
    self.effectiveColor = nil;
}
- (BOOL)prepareColorTransfer:(size_t)width height:(size_t)height {
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
            (id)kCVPixelBufferWidthKey: @(width), (id)kCVPixelBufferHeightKey: @(height),
            (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
            (id)kCVPixelBufferMetalCompatibilityKey: @YES
        };
        result = CVPixelBufferPoolCreate(NULL, NULL, (__bridge CFDictionaryRef)attributes, &_colorPool);
    }
    if (result == noErr) return YES;
    [self releaseColorTransfer];
    self.error = [NSString stringWithFormat:@"Cannot initialize SDR color conversion (%d)", (int)result];
    return NO;
}
- (void)consumeSample:(CMSampleBufferRef)sample clock:(CMClockRef)clock {
    CVPixelBufferRef pixels=CMSampleBufferGetImageBuffer(sample);
    if(!pixels) return;
    OSType format = CVPixelBufferGetPixelFormatType(pixels);
    if((format != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange &&
        format != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)) {
        self.error=@"Capture negotiated outside the required 1080p NV12 contract";
        return;
    }
    size_t width=CVPixelBufferGetWidth(pixels),height=CVPixelBufferGetHeight(pixels);
    if(!_colorPool||width!=_width||height!=_height) {
        [self releaseColorTransfer];
        if(![self prepareColorTransfer:width height:height])return;
        _width=width;_height=height;
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
        self.captureColor = color;
    }
    CVPixelBufferRef rgb = NULL;
    NSDictionary *limit = @{(id)kCVPixelBufferPoolAllocationThresholdKey: @6};
    OSStatus result = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(NULL, _colorPool,
        (__bridge CFDictionaryRef)limit, &rgb);
    if (result == kCVReturnWouldExceedAllocationThreshold) { self.drops++; return; }
    if (result == noErr) {
        for (NSUInteger i = 0; i < 3; i++)
            if (needsAttachment[i])
                CVBufferSetAttachment(pixels, keys[i], (__bridge CFTypeRef)effective[names[i]], kCVAttachmentMode_ShouldPropagate);
        result = VTPixelTransferSessionTransferImage(_colorTransfer, pixels, rgb);
        for (NSUInteger i = 0; i < 3; i++)
            if (needsAttachment[i]) CVBufferRemoveAttachment(pixels, keys[i]);
    }
    if (result == noErr) {
        self.frames++;self.frame(rgb,CMTimeGetSeconds(pts));
    } else {
        self.error = [NSString stringWithFormat:@"NV12 color conversion failed (%d)", (int)result];
    }
    if (rgb) CFRelease(rgb);
}

@end
