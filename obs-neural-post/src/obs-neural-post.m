// OBS native SDR HDMI source. IOSurface handoff follows mac-avcapture's fast path.
// GPL-2.0-or-later, matching OBS Studio. Core ML appearance is an opt-in experimental student; no Python host.
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>
#import <obs-module.h>
#import "neural-post.h"

OBS_DECLARE_MODULE()
MODULE_EXPORT const char *obs_module_description(void) { return "720p to 4K compact MLX neural post processing and native display presentation"; }

@interface NeuralCapture : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(strong) AVCaptureSession *session;
@property(strong) dispatch_queue_t captureQueue;
@property(strong) dispatch_queue_t controlQueue;
@property void *engine;
@property uint64_t sequence;
@property gs_texture_t *texture;
@property void *frameLease;
@property(copy) NSString *deviceID;
@property(copy) NSString *modelPath;
@property(copy) NSString *appearancePath;
@property(copy) NSString *anePath;
@property(copy) NSString *captureError;
- (void)configure:(obs_data_t *)settings;
- (void)shutdown;
@end

@implementation NeuralCapture
- (instancetype)init {
    if ((self=[super init])) {
        _captureQueue=dispatch_queue_create("obs.neural.capture",DISPATCH_QUEUE_SERIAL);
        _controlQueue=dispatch_queue_create("obs.neural.control",DISPATCH_QUEUE_SERIAL);
        _captureError=@"";
    }
    return self;
}
- (void)configure:(obs_data_t *)settings {
    NSString *deviceID=@(obs_data_get_string(settings,"device"));
    NSString *model=@(obs_data_get_string(settings,"model"));
    NSString *appearance=@(obs_data_get_string(settings,"appearance"));
    NSString *ane=@(obs_data_get_string(settings,"ane"));
    if ([_anePath isEqual:ane] && [_deviceID isEqual:deviceID] && [_modelPath isEqual:model] && [_appearancePath isEqual:appearance]) return;
    _deviceID=deviceID; _modelPath=model; _appearancePath=appearance; _anePath=ane;
    // All capture/inference-handle mutations are serialized here. Draining the
    // callback queue before teardown prevents a callback from using a retired engine.
    dispatch_sync(_controlQueue, ^{
        [self.session stopRunning]; self.session=nil;
        dispatch_sync(self.captureQueue, ^{});
        @synchronized(self) { np_destroy(self.engine); self.engine=NULL; }
        self.captureError=@"";
        if(ane.length && appearance.length) { self.captureError=@"Select ANE pilot or legacy appearance; stacking is unsupported"; return; }
        @synchronized(self) { self.engine=ane.length ? np_create_hybrid(model.UTF8String,ane.UTF8String) : np_create(model.UTF8String,appearance.UTF8String); }
        if (!self.engine || deviceID.length==0) return;
        AVCaptureDevice *device=[AVCaptureDevice deviceWithUniqueID:deviceID];
        if (!device) { self.captureError=@"Selected capture device is unavailable"; return; }
        NSError *error=nil;
        AVCaptureDeviceInput *input=[AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
        if (!input) { self.captureError=error.localizedDescription; return; }
        AVCaptureSession *session=[AVCaptureSession new];
        [session beginConfiguration];
        if (![session canAddInput:input]) { self.captureError=@"Cannot add capture input"; [session commitConfiguration]; return; }
        [session addInput:input];
        if ([session canSetSessionPreset:AVCaptureSessionPreset1280x720]) session.sessionPreset=AVCaptureSessionPreset1280x720;
        AVCaptureDeviceFormat *chosen=nil;
        AVFrameRateRange *chosenRange=nil;
        for (AVCaptureDeviceFormat *format in device.formats) {
            CMVideoDimensions d=CMVideoFormatDescriptionGetDimensions(format.formatDescription);
            if(d.width!=1280||d.height!=720) continue;
            for(AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
                if(range.maxFrameRate>=59.9 && range.minFrameRate<=60.01) {
                    chosen=format; chosenRange=range; break;
                }
            }
            if(chosen) break;
        }
        if(!chosen) { self.captureError=@"Device has no 720p59.94/60 mode"; [session commitConfiguration]; return; }
        AVCaptureVideoDataOutput *output=[AVCaptureVideoDataOutput new];
        output.alwaysDiscardsLateVideoFrames=YES;
        output.videoSettings=@{(id)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_32BGRA),
                              (id)kCVPixelBufferMetalCompatibilityKey:@YES,
                              (id)kCVPixelBufferWidthKey:@1280,(id)kCVPixelBufferHeightKey:@720};
        [output setSampleBufferDelegate:self queue:self.captureQueue];
        if(![session canAddOutput:output]) { self.captureError=@"Cannot add video output"; [session commitConfiguration]; return; }
        [session addOutput:output];
        // On macOS the output connection has its own cadence. Device format alone
        // left this USB card delivering 1080p20 in the real PS5 capture test.
        CMTime duration=chosenRange.minFrameDuration;
        AVCaptureConnection *connection=[output connectionWithMediaType:AVMediaTypeVideo];
        if(connection.supportsVideoMinFrameDuration) connection.videoMinFrameDuration=duration;
        if(connection.supportsVideoMaxFrameDuration) connection.videoMaxFrameDuration=duration;
        if(![device lockForConfiguration:&error]) { self.captureError=error.localizedDescription; [session commitConfiguration]; return; }
        device.activeFormat=chosen;
        device.activeVideoMinFrameDuration=duration; device.activeVideoMaxFrameDuration=duration;
        [device unlockForConfiguration];
        [session commitConfiguration];
        self.session=session; [session startRunning];
        // Keep the configured device format fixed while the stream is running.
        // Real-device relaunch tests exposed an intermittent capture stall when
        // activeFormat was written again immediately after startRunning.
    });
}
- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sample fromConnection:(AVCaptureConnection *)connection {
    (void)output; (void)connection;
    CVPixelBufferRef pixels=CMSampleBufferGetImageBuffer(sample);
    if(!pixels) return;
    if(CVPixelBufferGetWidth(pixels)!=1280 || CVPixelBufferGetHeight(pixels)!=720 ||
       CVPixelBufferGetPixelFormatType(pixels)!=kCVPixelFormatType_32BGRA) {
        self.captureError=@"Capture negotiated outside the required 720p BGRA contract";
        return;
    }
    CMTime pts=CMSampleBufferGetPresentationTimeStamp(sample);
    CMClockRef clock=self.session.synchronizationClock;
    if(clock && clock!=CMClockGetHostTimeClock()) pts=CMSyncConvertTime(pts,clock,CMClockGetHostTimeClock());
    if(!CMTIME_IS_NUMERIC(pts)) return;
    np_submit(_engine,(void *)pixels,++_sequence,CMTimeGetSeconds(pts));
}
- (void)shutdown {
    dispatch_sync(_controlQueue, ^{
        [self.session stopRunning]; self.session=nil;
        dispatch_sync(self.captureQueue, ^{});
        @synchronized(self) { np_destroy(self.engine); self.engine=NULL; }
    });
    if(_texture) { obs_enter_graphics(); gs_texture_destroy(_texture); _texture=NULL; obs_leave_graphics(); }
    np_frame_release(_frameLease); _frameLease=NULL;
}
@end

static const char *source_name(void *unused) { (void)unused; return "Neural HDMI · MLX + Metal (experimental)"; }
static void source_defaults(obs_data_t *settings) {
    NSString *binary=@(obs_get_module_binary_path(obs_current_module()));
    NSString *model=[[[binary stringByDeletingLastPathComponent] stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"Resources/rt4ksr-x3.safetensors"];
    obs_data_set_default_string(settings,"model",model.fileSystemRepresentation);
}
static void *source_create(obs_data_t *settings, obs_source_t *source) {
    (void)source;
    NeuralCapture *capture=[NeuralCapture new]; [capture configure:settings];
    return (__bridge_retained void *)capture;
}
static void source_destroy(void *data) {
    NeuralCapture *capture=CFBridgingRelease(data); [capture shutdown];
}
static void source_update(void *data,obs_data_t *settings) {
    [(__bridge NeuralCapture *)data configure:settings];
}
static void source_render(void *data,gs_effect_t *unused) {
    (void)unused;
    NeuralCapture *capture=(__bridge NeuralCapture *)data;
    void *frame=NULL;
    @synchronized(capture) { frame=np_frame_acquire(capture.engine); }
    if(!frame) return;
    CVPixelBufferRef pixels=np_frame_pixels(frame);
    IOSurfaceRef surface=CVPixelBufferGetIOSurface(pixels);
    // Fresh immutable output surfaces are not returned to a pixel-buffer pool.
    // Backend texture ownership therefore cannot race pooled buffer rewriting.
    if(!capture.texture) capture.texture=gs_texture_create_from_iosurface(surface);
    else if(frame!=capture.frameLease && !gs_texture_rebind_iosurface(capture.texture,surface)) {
        np_frame_release(frame); return;
    }
    np_frame_release(capture.frameLease); capture.frameLease=frame;
    if(!capture.texture) return;
    gs_effect_t *effect=obs_get_base_effect(gs_get_device_type()==GS_DEVICE_OPENGL ? OBS_EFFECT_DEFAULT_RECT : OBS_EFFECT_DEFAULT);
    bool previous=gs_framebuffer_srgb_enabled();
    bool linear=gs_get_linear_srgb(); gs_enable_framebuffer_srgb(linear);
    gs_eparam_t *param=gs_effect_get_param_by_name(effect,"image");
    if(linear) gs_effect_set_texture_srgb(param,capture.texture); else gs_effect_set_texture(param,capture.texture);
    while(gs_effect_loop(effect,"Draw")) gs_draw_sprite(capture.texture,0,3840,2160);
    gs_enable_framebuffer_srgb(previous);
}
static uint32_t source_width(void *data) { (void)data; return 3840; }
static uint32_t source_height(void *data) { (void)data; return 2160; }
static bool preview_clicked(obs_properties_t *p,obs_property_t *property,void *data) {
    (void)p; (void)property;
    NeuralCapture *capture=(__bridge NeuralCapture *)data;
    @synchronized(capture) { np_show_preview(capture.engine); }
    return false;
}
static obs_properties_t *source_properties(void *data) {
    obs_properties_t *p=obs_properties_create();
    obs_property_t *devices=obs_properties_add_list(p,"device","720p60 capture device",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(devices,"Select device","");
    AVCaptureDeviceDiscoverySession *discovery=[AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeExternal,AVCaptureDeviceTypeBuiltInWideAngleCamera] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
    for(AVCaptureDevice *device in discovery.devices) obs_property_list_add_string(devices,device.localizedName.UTF8String,device.uniqueID.UTF8String);
    obs_properties_add_path(p,"model","RT4KSR x3 MLX weights (empty = MPS spatial baseline)",OBS_PATH_FILE,"SafeTensors (*.safetensors)",NULL);
    obs_properties_add_path(p,"appearance","Trained appearance weights (optional)",OBS_PATH_FILE,"SafeTensors (*.safetensors)",NULL);
    obs_properties_add_path(p,"ane","Experimental teacher-trained ANE appearance model",OBS_PATH_DIRECTORY,NULL,NULL);
    obs_properties_add_button(p,"preview","Open native ProMotion preview (up to 120 Hz)",preview_clicked);
    char status[1024]={0};
    if(data) {
        NeuralCapture *capture=(__bridge NeuralCapture *)data;
        @synchronized(capture) { np_status(capture.engine,status,sizeof(status)); }
        if(capture.captureError.length) snprintf(status,sizeof(status),"Capture: %s",capture.captureError.UTF8String);
    }
    obs_properties_add_text(p,"status",status[0] ? status : "MLX reconstruction with optional Core ML appearance pilot. SDR video only; audio uses a separate OBS source.",OBS_TEXT_INFO);
    return p;
}
bool obs_module_load(void) {
    NSString *binary=@(obs_get_module_binary_path(obs_current_module()));
    NSString *metallib=[[[binary stringByDeletingLastPathComponent] stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"Resources/mlx.metallib"];
    if(![[NSFileManager defaultManager] fileExistsAtPath:metallib]) {
        blog(LOG_ERROR,"Neural Post: packaged MLX Metal library is missing"); return false;
    }
    np_set_metallib(metallib.fileSystemRepresentation);
    struct obs_source_info info={0};
    info.id="mac_neural_hdmi"; info.type=OBS_SOURCE_TYPE_INPUT;
    info.output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_CUSTOM_DRAW;
    info.get_name=source_name; info.create=source_create; info.destroy=source_destroy;
    info.update=source_update; info.get_properties=source_properties;
    info.get_defaults=source_defaults;
    info.get_width=source_width; info.get_height=source_height; info.video_render=source_render;
    info.icon_type=OBS_ICON_TYPE_CAMERA;
    obs_register_source(&info);
    return true;
}
