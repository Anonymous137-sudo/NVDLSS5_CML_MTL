// Independent normal-level window for testing click-through beneath the filter.
#import <AppKit/AppKit.h>
@interface Fixture : NSObject <NSApplicationDelegate>
@property NSWindow *window;
@property NSButton *button;
@property NSInteger clicks;
@property NSString *output;
@end
@implementation Fixture
- (void)applicationDidFinishLaunching:(NSNotification *)note {
 (void)note;NSRect screen=NSScreen.mainScreen.frame;
 self.window=[[NSWindow alloc] initWithContentRect:NSMakeRect(screen.size.width-570,350,520,320) styleMask:NSWindowStyleMaskTitled backing:NSBackingStoreBuffered defer:NO];
 self.window.title=@"Capture / click-through fixture";self.window.releasedWhenClosed=NO;
 self.button=[NSButton buttonWithTitle:@"Click-through test · 0" target:self action:@selector(clicked)];self.button.frame=NSMakeRect(90,110,340,100);[self.window.contentView addSubview:self.button];
 NSTextField *label=[NSTextField labelWithString:@"Known source window beneath the neural overlay"];label.frame=NSMakeRect(30,260,460,30);[self.window.contentView addSubview:label];
 [self.window makeKeyAndOrderFront:nil];[NSApp activateIgnoringOtherApps:YES];[self report];
}
- (void)report {
 NSRect r=[self.window convertRectToScreen:[self.button convertRect:self.button.bounds toView:nil]];
 NSDictionary *d=@{@"clicks":@(self.clicks),@"x":@(NSMidX(r)),@"y":@(NSScreen.mainScreen.frame.size.height-NSMidY(r)),@"windowNumber":@(self.window.windowNumber)};
 [[NSJSONSerialization dataWithJSONObject:d options:NSJSONWritingPrettyPrinted error:nil] writeToFile:self.output atomically:YES];
}
- (void)clicked {self.clicks++;self.button.title=[NSString stringWithFormat:@"Click-through test · %ld",(long)self.clicks];[self report];}
@end
int main(int argc,const char **argv){@autoreleasepool {if(argc!=2)return 2;NSApplication *app=NSApplication.sharedApplication;[app setActivationPolicy:NSApplicationActivationPolicyRegular];__attribute__((objc_precise_lifetime)) Fixture *fixture=[Fixture new];fixture.output=@(argv[1]);app.delegate=fixture;[app run];}return 0;}
