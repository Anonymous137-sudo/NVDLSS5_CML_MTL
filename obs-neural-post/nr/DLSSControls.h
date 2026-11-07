#pragma once
#import <AppKit/AppKit.h>
#import <obs.h>
void dlss_controls_show(obs_source_t *source, NSDictionary *(^statusProvider)(void));
void dlss_controls_close(void);
