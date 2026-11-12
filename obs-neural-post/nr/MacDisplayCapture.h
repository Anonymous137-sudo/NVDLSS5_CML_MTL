#pragma once
#import <Foundation/Foundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreGraphics/CoreGraphics.h>

// Start/stop belong to the supplied serial control queue. Samples are delivered
// on the existing capture queue; the block must consume/retain them before return.
@interface MacDisplayCapture : NSObject
@property BOOL nativeAspect;
@property(atomic, readonly) BOOL running;
@property(atomic, readonly) BOOL starting;
@property(atomic, readonly) CGDirectDisplayID displayID;
- (instancetype)initWithControlQueue:(dispatch_queue_t)controlQueue
                       captureQueue:(dispatch_queue_t)captureQueue
                              frame:(void (^)(CMSampleBufferRef))frame
                              error:(void (^)(NSString *))error;
- (void)startDisplay:(CGDirectDisplayID)displayID;
- (void)stop;
@end
