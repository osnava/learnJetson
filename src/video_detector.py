import cv2
from ultralytics import YOLO
import sys
import os

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

def process_video(input_video_path, output_video_path, model_path):
    """
    Process video with YOLO11n.engine and detect self-driving relevant objects

    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video with detections
        model_path: Path to YOLO11n.engine model
    """

    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

    # Release everything
    cap.release()
    out.release()

    print(f"\nDone! Output saved to: {output_video_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python video_detector.py <input_video> <output_video>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]
    model_path = "yolo11n.engine"

    if not os.path.exists(input_video):
        print(f"Error: Input video {input_video} does not exist")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} does not exist")
        sys.exit(1)

    process_video(input_video, output_video, model_path)
