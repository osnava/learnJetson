from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('/ssd/yolo11n.engine')

def generate_frames():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Flush camera buffer
    camera.release()
    camera = cv2.VideoCapture(0)

    # Warm up
    for _ in range(30):
        camera.read()

    while True:
        success, frame = camera.read()
        if not success:
            continue

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
        <head>
            <title>YOLO Detection</title>
        </head>
        <body style="margin:0; background:#000; display:flex; justify-content:center; align-items:center; height:100vh;">
            <img src="/video_feed" style="max-width:100%; max-height:100%;">
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
