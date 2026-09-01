import cv2
from ultralytics import YOLO
import sys
import os
import time

# Self-driving relevant categories from COCO dataset
SELFDRIVING_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    5: 'bus',
    3: 'motorcycle',
    7: 'truck',
    9: 'traffic light',
    11: 'stop sign'
}

# Hardware decode notes (issue #8, benchmarked 2026-09-01 — see
# docs/performance/nvdec-decode-notes.md):
# - Orin Nano has NVDEC (H.264/H.265 decode) but no NVENC, so the output
#   writer stays software (mp4v).
# - NVDEC is used when this OpenCV build supports in-process GStreamer
#   (cv2 CAP_GSTREAMER). The ultralytics container's pip OpenCV is built
#   WITHOUT GStreamer, so it falls back to software there — which benchmarks
#   FASTER than any subprocess bridge (software 1080p50 decode is ~180 fps;
#   a gst-launch pipe with NV12 costs more than it saves).
# - JETSON_NVDEC=0 forces software decode (benchmark baseline / debugging).
H264_FOURCCS = {'avc1', 'h264', 'x264', 'davc'}
HEVC_FOURCCS = {'hvc1', 'hev1', 'hevc', 'h265'}


def fourcc_str(value):
    value = int(value) & 0xFFFFFFFF
    return ''.join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def probe_video(path):
    """Read codec/dimensions via the software backend (header-only open)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    info = {
        'fourcc': fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)).lower(),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def build_gst_pipeline(path, parser):
    """GStreamer pipeline decoding on NVDEC, delivering BGR to an appsink."""
    return (
        f'filesrc location="{path}" ! qtdemux ! {parser} ! '
        'nvv4l2decoder enable-max-performance=1 ! nvvidconv '
        '! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR '
        '! appsink sync=false'
    )


def primed_source(cap, first):
    """Yield an already-read first frame, then everything cap produces.
    Closing the generator releases the capture (callers must close it)."""
    try:
        yield first
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def open_decoder(path, fourcc):
    """Return (mode, frames) — frames is a primed generator the caller must
    close — or (None, None) if the video cannot be decoded at all. Decode
    preference: NVDEC via cv2 GStreamer (when this OpenCV build has it),
    else software."""
    if os.environ.get('JETSON_NVDEC', '1') != '0':
        parser = None
        if fourcc in H264_FOURCCS:
            parser = 'h264parse'
        elif fourcc in HEVC_FOURCCS:
            parser = 'h265parse'
        if parser:
            try:
                cap = cv2.VideoCapture(
                    build_gst_pipeline(path, parser), cv2.CAP_GSTREAMER)
                ok, first = cap.read() if cap.isOpened() else (False, None)
                if ok:
                    return 'nvdec (cv2 GStreamer)', primed_source(cap, first)
                cap.release()
            except cv2.error:
                pass
            print('NVDEC (cv2 GStreamer) unavailable on this OpenCV build '
                  '— using software decode')

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    ok, first = cap.read()
    if not ok:
        cap.release()
        return None, None
    codec = fourcc if fourcc else 'unknown'
    return f'software ({codec})', primed_source(cap, first)


def process_video(input_video_path, output_video_path, model_path, max_frames=None):
    """
    Process video with YOLO11n.engine and detect self-driving relevant objects

    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video with detections
        model_path: Path to YOLO11n.engine model
        max_frames: Optional frame limit (quick previews / benchmarking)
    """

    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    info = probe_video(input_video_path)
    if info is None:
        print(f"Error: Cannot open video {input_video_path}")
        return

    print(f"Video info: {info['width']}x{info['height']} @ {info['fps']:.2f}fps, "
          f"{info['total_frames']} frames, codec {info['fourcc']!r}")

    mode, frames = open_decoder(input_video_path, info['fourcc'])
    if frames is None:
        print(f"Error: Cannot decode video {input_video_path}")
        return
    print(f"Decode: {mode}")

    # No hardware encoder on Orin Nano: the writer stays on the CPU (issue #8)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, info['fps'],
                          (info['width'], info['height']))

    frame_count = 0
    start = time.time()

    print("Processing video...")
    try:
        for frame in frames:
            frame_count += 1

            # Run inference
            results = model(frame, verbose=False)

            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class id
                    cls_id = int(box.cls[0])

                    # Filter only self-driving relevant classes
                    if cls_id in SELFDRIVING_CLASSES:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Get confidence
                        conf = float(box.conf[0])

                        # Get class name
                        class_name = SELFDRIVING_CLASSES[cls_id]

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                      (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Write frame
            out.write(frame)

            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{info['total_frames']} frames "
                      f"({frame_count/info['total_frames']*100:.1f}%)")

            if max_frames is not None and frame_count >= max_frames:
                print(f"Frame limit reached ({max_frames}) — stopping early")
                break
    finally:
        frames.close()  # releases the decoder deterministically
        out.release()

    elapsed = time.time() - start
    print(f"\nDone! {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.1f} fps end-to-end, decode: {mode})")
    print(f"Output saved to: {output_video_path}")

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python video_detector.py <input_video> <output_video> [max_frames]")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]
    max_frames = int(sys.argv[3]) if len(sys.argv) == 4 else None
    model_path = "yolo11n-int8.engine"  # INT8 (issue #7); FP16 fallback: yolo11n.engine

    if not os.path.exists(input_video):
        print(f"Error: Input video {input_video} does not exist")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} does not exist")
        sys.exit(1)

    process_video(input_video, output_video, model_path, max_frames)
