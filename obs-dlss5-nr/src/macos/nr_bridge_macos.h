// Per-filter state. No global model/history shared between OBS sources.
#pragma once
#include "../bridge/nr_bridge.h"
#include "nr_macos.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>

class NrBridgeSession {
	void *session = nullptr;
	std::string error;
	std::string timing;
	std::atomic<bool> diagnostics{false};
	std::atomic<bool> retry{false};

public:
	~NrBridgeSession()
	{
		if (session)
			nr5_destroy(session);
	}
	bool init(int, const char *directory, const char *)
	{
		session = nr5_create(directory);
		if (!session)
			error = "Cannot create the native Metal/MLX session";
		return session != nullptr;
	}
	bool ready() const { return session != nullptr; }
	void set_diagnostics(bool value) { diagnostics.store(value); }
	void request_retry() { retry.store(true); }
	const char *gpu_name() const { return "Apple Silicon · MLX FP16 / Metal"; }
	const char *last_error() const { return error.c_str(); }
	const char *timing_text() const { return timing.c_str(); }
	void detach_shared() {}
	bool process(const uint8_t *input, int input_pitch, uint8_t *output, int output_pitch, int width, int height,
		     const NrBridgeParams &p)
	{
		char settings[512];
		std::snprintf(settings, sizeof(settings),
			      "{\"style\":%d,\"preset\":%d,\"intensity\":%.9g,\"tone\":%.9g,"
			      "\"structure\":%.9g,\"skin\":%.9g,\"automask\":%s,\"temporal\":%s,"
			      "\"reset\":%s,\"mode\":%d,\"channelOrder\":%d}",
			      p.style, p.preset, p.intensity, p.tone, p.structure, p.skin,
			      p.automask ? "true" : "false", p.temporal ? "true" : "false",
			      (p.reset || retry.exchange(false)) ? "true" : "false", p.mode, p.channel_order);
		const auto started = std::chrono::steady_clock::now();
		bool ok = nr5_process(session, input, input_pitch, output, output_pitch, width, height, settings,
				      p.timestamp_ns);
		char message[1024]{};
		nr5_error(session, message, sizeof(message));
		error = message;
		const double ms =
			std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
		if (diagnostics.load()) {
			std::fprintf(stderr, "obs-dlss5-nr native: %dx%d, cascade %d, temporal %d, %.2f ms, %s\n",
				     width, height, p.mode, p.temporal, ms, ok ? "completed" : message);
		}
		return ok;
	}
};
