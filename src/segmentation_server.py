import os
import threading
import time

from flask import Flask, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO11n segmentation model (use TensorRT engine for better performance)
# Use 'yolo11n-seg.engine' if available, otherwise 'yolo11n-seg.pt'
model = YOLO('yolo11n-seg.engine')  # or 'yolo11n-seg.pt' if engine not yet created

# Knobs NOT set, deliberately (issue #9 "consider" item): imgsz/half are baked
# into the compiled TensorRT engines (changing them means rebuilding engines —
# see docs/performance/int8-tensorrt-engines.md), and vid_stride applies to
# video files, not live cameras. Frame skipping for slow consumers happens
# naturally: viewers always receive the latest annotated frame.

# Source: camera index (default 0) or a video file path (loops on EOF) —
# the file mode exists so the pipeline is testable/benchmarkable without a
# camera attached.
SOURCE = os.environ.get("SOURCE", "0")
IS_CAMERA = SOURCE.isdigit()

# One capture -> inference -> JPEG encode loop shared by every viewer
# (issue #9): inference cost is independent of viewer count, and the TRT
# engine is only ever called from this single thread.
_latest_jpeg = None
_cond = threading.Condition()
_stats = {"server_fps": 0.0, "viewers": 0, "last_frame_ts": 0.0}
_STALE_AFTER_S = 10  # viewers close their stream if no new frame for this long


def _open_source():
    if IS_CAMERA:
        cap = cv2.VideoCapture(int(SOURCE))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(SOURCE)


def _capture_loop():
    global _latest_jpeg
    cap = _open_source()
    for _ in range(30):  # warm up
        cap.read()
    frames = 0
    t0 = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            if IS_CAMERA:
                time.sleep(0.01)
                continue
            cap.release()  # EOF: loop the file
            cap = _open_source()
            continue

        # Run segmentation inference
        results = next(model(frame, stream=True, verbose=False))

        # Plot results with masks
        annotated_frame = results.plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        with _cond:
            _latest_jpeg = buffer.tobytes()
            _stats["last_frame_ts"] = time.time()
            _cond.notify_all()

        frames += 1
        now = time.time()
        if now - t0 >= 5:
            _stats["server_fps"] = round(frames / (now - t0), 1)
            frames = 0
            t0 = now


def _stream_frames():
    last_sent = None
    with _cond:
        _stats["viewers"] += 1
    try:
        while True:
            with _cond:
                _cond.wait_for(
                    lambda: _latest_jpeg is not None and _latest_jpeg is not last_sent,
                    timeout=_STALE_AFTER_S)
                if _latest_jpeg is None or _latest_jpeg is last_sent:
                    if time.time() - _stats["last_frame_ts"] > _STALE_AFTER_S:
                        return  # capture loop stalled or died — end the stream
                    continue
                jpeg = _latest_jpeg
            last_sent = jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
    finally:
        with _cond:
            _stats["viewers"] -= 1


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
        <head>
            <title>YOLO Segmentation</title>
        </head>
        <body style="margin:0; background:#000; display:flex; flex-direction:column; justify-content:center; align-items:center; height:100vh;">
            <h1 style="color:#fff; font-family:Arial;">YOLO11n Instance Segmentation</h1>
            <img src="/video_feed" style="max-width:90%; max-height:80%; border:2px solid #fff;">
        </body>
    </html>
    '''


@app.route('/video_feed')
def video_feed():
    return Response(_stream_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    return jsonify(source=SOURCE, **_stats)

if __name__ == '__main__':
    threading.Thread(target=_capture_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)
