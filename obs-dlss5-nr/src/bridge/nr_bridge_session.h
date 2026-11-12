#pragma once
#ifdef __APPLE__
#include "../macos/nr_bridge_macos.h"
#else
#include "nr_bridge.h"
// Preserve the Windows bridge and driver lifetime policy.
class NrBridgeSession {
public:
	bool init(int gpu, const wchar_t *runtime, const wchar_t *shim) { return nrbridge::init(gpu, runtime, shim); }
	bool ready() const { return nrbridge::ready(); }
	void set_diagnostics(bool value) { nrbridge::set_diagnostics(value); }
	void request_retry() { nrbridge::request_retry(); }
	const char *gpu_name() const { return nrbridge::gpu_name(); }
	const char *last_error() const { return nrbridge::last_error(); }
	const char *timing_text() const { return nrbridge::timing_text(); }
	void detach_shared() { nrbridge::detach_shared(); }
	bool attach_shared(uint32_t input, uint32_t w, uint32_t h, uint32_t *output)
	{
		return nrbridge::attach_shared(input, w, h, output);
	}
	bool process_gpu(int w, int h, const NrBridgeParams &p) { return nrbridge::process_gpu(w, h, p); }
	bool process(const uint8_t *input, int input_pitch, uint8_t *output, int output_pitch, int w, int h,
		     const NrBridgeParams &p)
	{
		return nrbridge::process(input, input_pitch, output, output_pitch, w, h, p);
	}
};
#endif
