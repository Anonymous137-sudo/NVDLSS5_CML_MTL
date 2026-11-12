// Run against an installed OBS runtime, using a synthetic animated 4K source.
// No user's scene collection or capture device is opened or changed.
#include <obs.h>
#include <graphics/vec4.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

static std::atomic<uint64_t> received{0}, changed{0}, nonblack{0};
static uint64_t previous_hash = 0;

static const char *source_name(void *)
{
	return "Performance test pattern";
}
static void *source_create(obs_data_t *, obs_source_t *)
{
	return new int(0);
}
static void source_destroy(void *p)
{
	delete static_cast<int *>(p);
}
static uint32_t source_width(void *)
{
	return 320;
}
static uint32_t source_height(void *)
{
	return 320;
}
static void source_render(void *, gs_effect_t *)
{
	const uint64_t tick = obs_get_video_frame_time() / 33333333;
	vec4 color;
	vec4_set(&color, 0.15f + float(tick % 40) / 80.0f, 0.35f, 0.65f, 1.0f);
	gs_effect_t *effect = obs_get_base_effect(OBS_EFFECT_SOLID);
	gs_effect_set_vec4(gs_effect_get_param_by_name(effect, "color"), &color);
	while (gs_effect_loop(effect, "Solid"))
		gs_draw_sprite(nullptr, 0, 320, 320);
	vec4_set(&color, 0.75f, 0.2f + float(tick % 30) / 60.0f, 0.2f, 1.0f);
	gs_effect_set_vec4(gs_effect_get_param_by_name(effect, "color"), &color);
	while (gs_effect_loop(effect, "Solid"))
		gs_draw_sprite(nullptr, 0, 160, 160);
}

static void receive_frame(void *, video_data *frame)
{
	uint64_t hash = 1469598103934665603ULL;
	uint64_t light = 0;
	for (uint32_t y = 0; y < 320; y += 47) {
		const uint8_t *row = frame->data[0] + y * frame->linesize[0];
		for (uint32_t x = 0; x < 320; x += 43) {
			for (uint32_t c = 0; c < 3; ++c) {
				hash = (hash ^ row[x * 4 + c]) * 1099511628211ULL;
				light += row[x * 4 + c];
			}
		}
	}
	++received;
	if (light > 0)
		++nonblack;
	if (hash != previous_hash)
		++changed;
	previous_hash = hash;
}

int main(int argc, char **argv)
{
	if (argc != 3)
		return 2;
	if (!obs_startup("en-US", nullptr, nullptr))
		return 3;
	obs_add_data_path("/Applications/OBS.app/Contents/Frameworks/libobs.framework/Resources");
	obs_video_info video{};
	video.graphics_module = "/Applications/OBS.app/Contents/Frameworks/libobs-metal.dylib";
	video.fps_num = 30;
	video.fps_den = 1;
	video.base_width = video.base_height = 320;
	video.output_width = video.output_height = 320;
	video.output_format = VIDEO_FORMAT_BGRA;
	video.colorspace = VIDEO_CS_709;
	video.range = VIDEO_RANGE_FULL;
	video.scale_type = OBS_SCALE_BILINEAR;
	if (obs_reset_video(&video) != OBS_VIDEO_SUCCESS)
		return 4;
	obs_module_t *module = nullptr;
	if (obs_open_module(&module, argv[1], argv[2]) != MODULE_SUCCESS || !obs_init_module(module))
		return 5;
	obs_source_info source{};
	source.id = "nr-native-pattern";
	source.type = OBS_SOURCE_TYPE_INPUT;
	source.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW;
	source.get_name = source_name;
	source.create = source_create;
	source.destroy = source_destroy;
	source.get_width = source_width;
	source.get_height = source_height;
	source.video_render = source_render;
	obs_register_source(&source);
	auto *pattern = obs_source_create_private(source.id, "native pattern", nullptr);
	auto *settings = obs_data_create();
	obs_data_set_int(settings, "cascade_mode", 1);
	obs_data_set_int(settings, "processing_height", 0);
	obs_data_set_int(settings, "nr_fps", 30);
	obs_data_set_bool(settings, "diagnostics", true);
	auto *filter = obs_source_create_private("obs-dlss5-nr-filter", "native NR", settings);
	if (!pattern || !filter)
		return 6;
	obs_source_filter_add(pattern, filter);
	obs_set_output_source(0, pattern);
	obs_add_raw_video_callback(nullptr, receive_frame, nullptr);
	std::this_thread::sleep_for(std::chrono::seconds(9));
	auto *properties = obs_source_properties(filter);
	obs_properties_destroy(properties);
	auto *state = obs_source_get_settings(filter);
	std::string status = obs_data_get_string(state, "status");
	bool enhanced = status.find("enhanced") != std::string::npos;
	std::printf("NR_MACOS_FILTER frames=%llu changed=%llu nonblack=%llu status=%s\n", received.load(),
		    changed.load(), nonblack.load(), status.c_str());
	obs_data_release(state);
	obs_remove_raw_video_callback(receive_frame, nullptr);
	obs_set_output_source(0, nullptr);
	obs_source_release(pattern);
	obs_source_release(filter);
	obs_data_release(settings);
	obs_shutdown();
	return enhanced && changed > 5 && nonblack > 5 ? 0 : 7;
}
