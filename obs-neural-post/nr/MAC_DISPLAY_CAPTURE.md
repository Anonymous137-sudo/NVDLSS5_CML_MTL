# Mac display input

Open **OBS → Tools → DLSS 5 → Live tools**. Set **Input source** to **Mac
display**, then choose a display under **Mac display**. Automatic selects an
available built-in display, otherwise the primary display. Use **Open native
ProMotion preview** to view the processed output. The same selectors are in the
ordinary OBS source properties.

The input selector changes capture, independently of the DLSS quality mode.
Fidelity, Balanced, Performance, Single-71 and Double-31 remain available with
the same controls. HDMI remains a selectable input for a connected console.

The native ScreenCaptureKit path requests 1920x1080 NV12 full-range SDR, BT.709
matrix, up to 60 capture frames/s, with a three-surface capture queue. It
preserves the source aspect ratio rather than cropping a taller Mac display.
It excludes OBS's own windows to prevent a recursive preview. The source uses
the existing bounded capture, colour conversion, native NR, MFG and presentation
path. It does not launch a script or a second viewer/capture engine.

If macOS blocks capture, use **Screen recording permission / reconnect display**
and allow OBS under **System Settings → Privacy & Security → Screen & System
Audio Recording**. This is separate from the camera permission for HDMI. On
systems that require an OBS restart after permission changes, reopen OBS and
select Mac display again. This machine already granted permission during the
live validation.

Changing the capture input/display stops the previous capture producer, drains
its sample callback queue and invalidates pending presentation/MFG work. The
next NR request explicitly resets model history. This keeps history from one
input out of the other without discarding the quality controls or preview.
Asynchronous ScreenCaptureKit startup uses a generation check so a late start
cannot replace a newer input selection or restart after shutdown.

## Validation on this Mac

The plugin and native runtime built and loaded in stock OBS. Live desktop
capture reported full-range NV12 and declared/effective BT.709 metadata. The
native preview showed the browser desktop and image content, with OBS windows
excluded. Selecting the built-in display reconnected capture successfully;
starts advanced from 3 to 4 and processed frames advanced from 482 to 574 in
the saved samples, with zero reported NR/MFG failures and no capture error.
The first display session also shut down cleanly in OBS's log.

Evidence is under `build/display-capture-live/`, including the reconnect
snapshots and `reconnect/mac-preview.png`. The previous installed module and
runtime libraries are backed up under `before-display/` in that directory.
The final selected input is Mac display. A console gameplay test is not part
of this validation while the PS5 is off.

This validates capture, reconnect and frame delivery; it does not establish
moving-scene temporal quality or a playable full-resolution rate. Fidelity
remained roughly 2-3 processed base frames/s during this session, and its long
frame gaps prevented RIFE intermediates. The 2160p processing setting remains
a maximum; the current captured/model input is 1920x1080. ProMotion display Hz
and fresh shown/generated frame rates remain separate measurements.
