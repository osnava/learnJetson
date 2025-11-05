from ultralytics import YOLO
import cv2
import sys

model = YOLO('racetrack_model.engine', task='segment')
video_path = sys.argv[1] if len(sys.argv) > 1 else 'input.mp4'
output_path = 'output_segmented.mp4'

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing {video_path}...")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[10, 3], verbose=False)  # racetrack, ego_vehicle
    out.write(results[0].plot())

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
out.release()
print(f"Done! Output: {output_path}")
