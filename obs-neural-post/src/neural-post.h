#pragma once
#include <stdint.h>
void np_set_metallib(const char *path);
void *np_create(const char *model_path, const char *appearance_path);
void *np_create_hybrid(const char *model_path, const char *ane_path);
void np_destroy(void *handle);
void np_submit(void *handle, void *pixel_buffer, uint64_t sequence, double host_seconds);
void np_reset(void *handle);
void *np_frame_acquire(void *handle);
void *np_frame_pixels(void *frame);
void np_frame_release(void *frame);
void np_show_preview(void *handle);
void np_status(void *handle, char *text, int32_t count);
