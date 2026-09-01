# NVDEC hardware decode in video_detector.py — findings (issue #8)

> **Benchmarked:** 2026-09-01, Jetson Orin Nano 8GB Super, JetPack 6.2.2 /
> TensorRT 10.3, `ultralytics/ultralytics:latest-jetson-jetpack6`
> (ultralytics 8.3.225, OpenCV 4.11.0 pip build), clocks locked at MAXN
> (issue #6), INT8 yolo11n engine (issue #7). Full logs on the Jetson at
> `/ssd/int8/issue8/` (runs, CPU samples, decode-only probes).

## TL;DR

The Orin Nano **has** NVDEC (H.264/H.265 hardware decode) but this stack
**cannot use it profitably**: software decode of 1080p50 H.264 already runs at
~180 fps (~0.2 CPU cores), while every zero-install bridge from GStreamer to
this OpenCV costs more than the decode it offloads. `video_detector.py`
therefore uses NVDEC **only** where OpenCV itself has GStreamer support
(in-process `cv2.CAP_GSTREAMER`) and falls back to software elsewhere.

## Why the obvious route is blocked

The issue's plan — `cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)` —
requires an OpenCV built **with GStreamer**. The ultralytics container's
OpenCV is a pip wheel: `cv2.getBuildInformation()` reports `GStreamer: NO`
(FFMPEG: YES). `gst-launch-1.0` and `nvv4l2decoder` *are* in the image; only
the cv2 binding is missing.

## Numbers

End-to-end (`video_detector.py`, model inference + drawing + mp4v write;
CPU = whole-system busy % across 6 cores, sampled at 2 Hz):

| Video | Decode | fps | Avg CPU |
|---|---|---|---|
| ny_streets.mp4 (640×360@30) | software | 33.4 | 16.7 % |
| ny_streets.mp4 | NVDEC via `gst-launch` pipe (BGRx) | 31.6 | 16.8 % |
| lewisf1lap.mp4 (1920×1080@50) | software | 19.2 | 19.4 % |
| lewisf1lap.mp4 | NVDEC via pipe (BGRx) | 13.3 | 17.4 % |

Decode-only comparison at 1080p (no inference/writer):

| Path | fps |
|---|---|
| cv2 FFMPEG (software) | **179.8** |
| NVDEC pipe, BGRx out (8.3 MB/frame) | 31.6 |
| NVDEC pipe, NV12 out (3.1 MB/frame) + `cv2.cvtColor` | 122.3 |

Pixel agreement between NVDEC and software frames: mean abs diff ~1.6/255
(YUV→RGB rounding only) — decode correctness was never the problem.

## Interpretation

1. **The premise (decode competes with inference for CPU) barely holds.**
   Going from 360p to 1080p (4.7× the pixels) raised system CPU only
   16.7 % → 19.4 %; ffmpeg's NEON decode is 3.6× realtime at 1080p50. The
   frame budget is dominated by inference + preprocessing + mp4v encoding
   (which has no hardware block to move to — no NVENC on this module).
2. **Subprocess pipes lose by construction here.** Linux pipes move 64 KB
   chunks by default: an 8.3 MB BGRx frame is ~128 blocking write/read
   pairs; NV12 (3.1 MB) cuts that 2.7× and is 4× faster than the BGRx pipe —
   but still 32 % slower than just letting FFMPEG decode on the CPU.
3. **The only route that could win is in-process** GStreamer decode
   (buffers flow through the pipeline while python infers, no pipe, and
   `nvvidconv` does the colorspace on the VIC). That requires an OpenCV
   with GStreamer support.

## Retry conditions (any one)

- A GStreamer-enabled OpenCV in the ultralytics container (system
  `python3-opencv` from the L4T apt feed, a jetson-built wheel, or a custom
  image) — `video_detector.py` already auto-selects NVDEC the moment
  `cv2.CAP_GSTREAMER` works; benchmark against `JETSON_NVDEC=0`.
- DeepStream (full NVMM pipeline, zero copy — but a different architecture
  and a much larger dependency).
- A workload with many concurrent decode streams (e.g. multi-camera), where
  aggregate software decode cost actually saturates the CPU — not this
  single-stream use case.

Benchmarking helpers live on the Jetson in `/ssd/int8/issue8/`
(`bench-issue8.sh`, `cpu_sampler.py`, `summarize.py`; decode-only probes in
`/ssd/int8/issue8-probe*.py`). `JETSON_NVDEC=0` forces software decode.
