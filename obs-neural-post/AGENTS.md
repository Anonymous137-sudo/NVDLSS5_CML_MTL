# Project workflow

- Validate runtime changes through the live PS5 HDMI pipeline. The user explicitly requires live-pipeline testing going forward. Trace/export inputs and focused mathematical checks may assist implementation, but repeated single-image timing is not the performance acceptance test.
- Preserve the original 720p appearance model and all weights unless the user changes that requirement. Lower-resolution appearance processing and tiny students are historical experiments, not the accepted quality target.
- Report fresh processed frames, generated frames and display repetitions separately. A request for 120 Hz presentation does not establish 120 fresh frames/s.
- Label GPU control frames during Core ML compilation explicitly. Selecting CPU+ANE excludes GPU inference but allows CPU fallback; preferred-device counts are not a hardware-utilization trace.
- Keep one bounded capture/inference path. Do not silently open duplicate HDMI sessions or replace the normal OBS plugin with a diagnostic runtime.
- State which model stages actually ran on each device, whether paired live-frame numerical checks passed, and whether the complete 30–45 fresh-FPS target passed.
