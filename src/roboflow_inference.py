from inference_sdk import InferenceHTTPClient
import cv2
import supervision as sv
import time

API_KEY = "yourAPIkey"
MODEL_ID = "f1onboard-vpltc/3"
INPUT_VIDEO = "lasvegasnor4.mp4"
OUTPUT_VIDEO = "lasvegasnor4_segmented.mp4"

print("=" * 50)
print("Starting video processing...")
print("=" * 50)

# Connect to local inference server
print(f"\n[1/5] Connecting to inference server at localhost:9001...")
try:
    client = InferenceHTTPClient(
        api_url="http://localhost:9001",
        api_key=API_KEY
    )
    print("✓ Connected successfully")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    exit(1)

# Open video
print(f"\n[2/5] Opening video: {INPUT_VIDEO}")
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"✗ Failed to open video file")
    exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"✓ Video opened successfully")
print(f"  - Resolution: {width}x{height}")
print(f"  - FPS: {fps}")
print(f"  - Total frames: {total_frames}")

# Setup output
print(f"\n[3/5] Setting up output: {OUTPUT_VIDEO}")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
if not out.isOpened():
    print(f"✗ Failed to create output video")
    exit(1)
print("✓ Output video ready")

# Annotators
print(f"\n[4/5] Initializing annotators...")
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()
print("✓ Annotators ready")

print(f"\n[5/5] Processing frames...")
print("-" * 50)

frame_count = 0
start_time = time.time()
last_update = start_time

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        # Run inference
        result = client.infer(frame, model_id=MODEL_ID)
        
        # Convert to supervision format
        detections = sv.Detections.from_inference(result)
        
        # Annotate
        annotated = mask_annotator.annotate(frame.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections)
        
        out.write(annotated)
        frame_count += 1
        
        # Progress update every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            progress = (frame_count / total_frames) * 100
            eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
            
            print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"Detections: {len(detections)} | "
                  f"Speed: {fps_actual:.1f} fps | "
                  f"ETA: {eta:.1f}s")
    
    except Exception as e:
        print(f"✗ Error at frame {frame_count}: {e}")
        continue

cap.release()
out.release()

total_time = time.time() - start_time
print("-" * 50)
print(f"\n✓ Processing complete!")
print(f"  - Total frames: {frame_count}")
print(f"  - Total time: {total_time:.2f}s")
print(f"  - Average speed: {frame_count/total_time:.2f} fps")
print(f"  - Output saved to: {OUTPUT_VIDEO}")
print("=" * 50)