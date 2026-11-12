#import "MacDisplayCapture.h"
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <CoreVideo/CoreVideo.h>
#import <unistd.h>

@interface MacDisplayCapture () <SCStreamOutput,SCStreamDelegate>
@property(atomic, readwrite) BOOL running;
@property(atomic, readwrite) BOOL starting;
@property(atomic, readwrite) CGDirectDisplayID displayID;
@property(atomic, strong) SCStream *stream;
@property(strong) dispatch_queue_t controlQueue,captureQueue;
@property(copy) void (^frame)(CMSampleBufferRef);
@property(copy) void (^reportError)(NSString *);
@property NSUInteger generation;
@end
@implementation MacDisplayCapture
- (instancetype)initWithControlQueue:(dispatch_queue_t)controlQueue
                       captureQueue:(dispatch_queue_t)captureQueue
                              frame:(void (^)(CMSampleBufferRef))frame
                              error:(void (^)(NSString *))error {
    if((self=[super init])){_controlQueue=controlQueue;_captureQueue=captureQueue;_frame=[frame copy];_reportError=[error copy];}
    return self;
}
- (void)startDisplay:(CGDirectDisplayID)requested {
    [self stop];
    if(!CGPreflightScreenCaptureAccess()) {
        self.reportError(@"Allow OBS in System Settings → Privacy & Security → Screen & System Audio Recording, then reconnect Mac display capture.");
        return;
    }
    self.starting=YES;
    NSUInteger generation=self.generation;
    [SCShareableContent getShareableContentExcludingDesktopWindows:YES onScreenWindowsOnly:NO
      completionHandler:^(SCShareableContent *content,NSError *error) {
        dispatch_async(self.controlQueue,^{
            if(generation!=self.generation)return;
            if(error){self.starting=NO;self.reportError(error.localizedDescription);return;}
            SCDisplay *chosen=nil;
            for(SCDisplay *display in content.displays) {
                if((requested&&display.displayID==requested)||(!requested&&CGDisplayIsBuiltin(display.displayID))){chosen=display;break;}
            }
            if(!chosen&&!requested)for(SCDisplay *display in content.displays)
                if(display.displayID==CGMainDisplayID()){chosen=display;break;}
            if(!chosen){self.starting=NO;self.reportError(@"Selected Mac display is unavailable. Select a connected display.");return;}
            NSMutableArray *excluded=[NSMutableArray array];
            for(SCRunningApplication *app in content.applications)if(app.processID==getpid())[excluded addObject:app];
            SCContentFilter *filter=[[SCContentFilter alloc] initWithDisplay:chosen excludingApplications:excluded exceptingWindows:@[]];
            SCStreamConfiguration *config=[SCStreamConfiguration new];
            config.width=1920;config.height=1080;
            if(self.nativeAspect) {
                double aspect=(double)CGDisplayPixelsWide(chosen.displayID)/MAX(1,CGDisplayPixelsHigh(chosen.displayID));
                config.height=(size_t)(MIN(1080.0,1920.0/aspect)/2)*2;
                config.width=(size_t)(config.height*aspect/2)*2;
            }
            config.minimumFrameInterval=CMTimeMake(1,60);
            config.pixelFormat=kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
            config.colorSpaceName=kCGColorSpaceSRGB;config.colorMatrix=kCGDisplayStreamYCbCrMatrix_ITU_R_709_2;
            config.queueDepth=3;config.showsCursor=!self.nativeAspect;config.capturesAudio=NO;
            config.scalesToFit=YES;config.preservesAspectRatio=YES;
            config.streamName=@"DLSS 5 Mac display input";
            if(@available(macOS 15.0,*))config.captureDynamicRange=SCCaptureDynamicRangeSDR;
            SCStream *stream=[[SCStream alloc] initWithFilter:filter configuration:config delegate:self];
            NSError *attachError=nil;
            if(![stream addStreamOutput:self type:SCStreamOutputTypeScreen sampleHandlerQueue:self.captureQueue error:&attachError]) {
                self.starting=NO;self.reportError(attachError.localizedDescription ?: @"Cannot attach display capture output");return;
            }
            self.displayID=chosen.displayID;self.stream=stream;
            [stream startCaptureWithCompletionHandler:^(NSError *startError){
                dispatch_async(self.controlQueue,^{
                    if(generation!=self.generation||self.stream!=stream){[stream stopCaptureWithCompletionHandler:nil];return;}
                    self.starting=NO;self.running=startError==nil;
                    if(startError){self.stream=nil;self.reportError(startError.localizedDescription);}
                });
            }];
        });
    }];
}
- (void)stop {
    self.generation++;self.running=NO;self.starting=NO;
    SCStream *stream=self.stream;self.stream=nil;
    [stream stopCaptureWithCompletionHandler:nil];
}
- (void)stream:(SCStream *)stream didOutputSampleBuffer:(CMSampleBufferRef)sample ofType:(SCStreamOutputType)type {
    if(stream!=self.stream||!self.running||type!=SCStreamOutputTypeScreen||!CMSampleBufferIsValid(sample))return;
    CFArrayRef attachments=CMSampleBufferGetSampleAttachmentsArray(sample,false);
    if(!attachments||CFArrayGetCount(attachments)==0)return;
    NSDictionary *info=(__bridge NSDictionary *)CFArrayGetValueAtIndex(attachments,0);
    NSNumber *status=info[SCStreamFrameInfoStatus];
    if(!status||status.integerValue!=SCFrameStatusComplete)return;
    self.frame(sample);
}
- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
    dispatch_async(self.controlQueue,^{
        if(stream!=self.stream)return;
        self.stream=nil;self.running=NO;self.starting=NO;self.reportError(error.localizedDescription);
    });
}
@end
