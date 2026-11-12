// Native ANE-only interface for offline ANEForge MIL + weights directories.
// Calls for a program must be serialized on its owner worker. Bind all surfaces
// before first execute, and keep them alive until the program is destroyed.
#pragma once
#include <IOSurface/IOSurfaceRef.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void *dlss_ane_compile_directory(const char *directory, const char *input_name,
                                 const char *output_name, size_t input_bytes,
                                 size_t output_bytes);
void *dlss_ane_compile_directory_inputs(const char *directory,
                                        const char *const *input_names,
                                        const size_t *input_bytes, size_t input_count,
                                        const char *output_name, size_t output_bytes);
void *dlss_ane_compile_directory_ports(const char *directory,
                                       const char *const *input_names, const size_t *input_bytes, size_t input_count,
                                       const char *const *output_names, const size_t *output_bytes, size_t output_count);
int dlss_ane_execute(void *program);
void dlss_ane_destroy(void *program);
// Requires tightly packed FP16 ports with a 64-byte-aligned final dimension.
// Returns zero only after IOSurface ID and mapped-address identity both pass.
int dlss_af_bind_surface(void *program, const char *name, int input, IOSurfaceRef surface);
// Captured FFN feasibility API. Borrows both programs; caller must serialize
// execution and retain the programs until context destruction. The graph ports
// must match the fixed 1024->4096->1024 checkpoint contract. Executes on an
// inference worker only: includes a dependent Metal completion wait. This is
// not installed in the production renderer. Timings contain expansion wall,
// gate handoff wall, branch wall, full wall and gate GPU time (milliseconds).
void *dlss_af_hybrid_create(void *expand, void *branch,
                            const char *expand_input, const char *expand_output,
                            const char *branch_input, const char *branch_skip,
                            const char *branch_output, int tokens);
// Same pipeline, with branch residual and following QKV in one ANE program.
// Both mid and QKV outputs are retained separately for the original attention.
void *dlss_af_hybrid_create_qkv(void *expand, void *branch,
                                const char *expand_input, const char *expand_output,
                                const char *branch_input, const char *branch_skip,
                                const char *branch_output, const char *qkv_output, int tokens);
int dlss_af_hybrid_execute(void *context, double timings[5]);
// GPU-resident, contiguous NHWC MTLBuffer source/destination (borrowed).
// Null destination skips unpack; use the retained output surface view instead.
// Source producer must be complete. Buffers and source MLX array must remain
// alive and immutable throughout execution; output consumer starts after return.
// Timings: full, pack wall, inner wall, unpack wall, pack GPU, gate GPU,
// unpack GPU, expansion ANE wall, branch ANE wall. Worker-only synchronous API.
int dlss_af_hybrid_execute_buffers(void *context, void *source_metal_buffer,
                                    void *destination_metal_buffer, double timings[9]);
// +1 surface ownership permits an MLX NCHW view without GPU unpack. Caller
// must finish every consumer before this context's next execution overwrites it.
IOSurfaceRef dlss_af_hybrid_copy_output_surface(void *context) CF_RETURNS_RETAINED;
IOSurfaceRef dlss_af_hybrid_copy_qkv_surface(void *context) CF_RETURNS_RETAINED;
void dlss_af_hybrid_destroy(void *context);
#ifdef __cplusplus
}
#endif
