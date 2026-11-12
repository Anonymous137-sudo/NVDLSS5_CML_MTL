// Native backend ABI for obs-dlss5-nr. GPL-2.0-or-later.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void *nr5_create(const char *resources);
void nr5_destroy(void *session);
bool nr5_process(void *session, const uint8_t *input, int input_pitch, uint8_t *output, int output_pitch, int width,
		 int height, const char *settings, uint64_t timestamp_ns);
void nr5_error(void *session, char *message, size_t capacity);
void nr5_reset(void *session);
// Import, flow, pass1, pass2, export, history-reset flag, input-copy, output-copy.
size_t nr5_profile(void *session, double *values, size_t capacity);
// CVPixelBufferRef in/out; returned buffer is retained (+1), caller releases it.
void *nr5_process_surface(void *session, void *input_surface, const char *settings, uint64_t timestamp_ns);
#ifdef __cplusplus
}
#endif
