#pragma once
#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreGraphics/CoreGraphics.h>
@interface NativeInput : NSObject
@property(atomic, copy, readonly) NSString *error;
- (instancetype)initWithFrame:(void (^)(CVPixelBufferRef,double))frame;
- (void)startScreen:(CGDirectDisplayID)display;
- (void)startHDMI:(NSString *)device;
- (void)stop;
- (NSDictionary *)status;
@end
