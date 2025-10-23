# Jetson Orin Nano AI Development

This repository contains AI/ML development projects and deployment configurations for the NVIDIA Jetson Orin Nano, including vision processing, LLM inference, and self-driving applications.

## Prerequisites

- NVIDIA Jetson Orin Nano (8GB) with JetPack 6
- Docker with NVIDIA runtime support
- NVMe SSD mounted at `/ssd` (highly recommended for performance and storage)

## Jetson Configuration

### Finding Jetson IP Address

To connect via SSH, you need to find the Jetson's IP address:

```bash
# Show all network interfaces and their IPs
ip addr show

# Or use hostname -I to show all IPs
hostname -I
```

**Common interfaces:**
- `eth0`: Ethernet connection
- `wlan0`: WiFi connection
- `l4tbr0` or `usb0`: USB network connection (when connected via USB to host computer)

**Quick reference:**
```bash
# Ethernet IP
ip addr show eth0 | grep 'inet '

# WiFi IP
ip addr show wlan0 | grep 'inet '

# USB network IP
ip addr show l4tbr0 | grep 'inet '
```

### Disable GUI to Free GPU Memory

For better GPU performance, disable the graphical interface and use SSH-only mode:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

To re-enable the GUI later (if needed):

```bash
sudo systemctl set-default graphical.target
sudo reboot
```

### Performance Optimization

Best practices for maximizing performance on the Jetson Orin Nano:

#### Enable MAX Power Mode

Ensures all CPU and GPU cores are turned on:

```bash
sudo nvpmodel -m 2
```

#### Enable Jetson Clocks

Clocks all CPU and GPU cores at their maximum frequency:

```bash
sudo jetson_clocks
```

#### Install Jetson Stats Application

Monitor system temperatures, CPU/GPU/RAM utilization, and manage performance settings:

```bash
sudo apt update
sudo pip install jetson-stats
sudo reboot
```

Run the monitoring tool:

```bash
jtop
```

**Note:** These optimizations are especially important when running compute-intensive workloads like YOLO object detection or LLM inference.

## Setup

This repository supports multiple AI/ML workloads. Choose the setup that matches your use case:

### 1. Vision Processing (Object Detection)

Uses the Ultralytics container for YOLO-based object detection and vision tasks.

**Demo:**

![Object Detection Demo](resources/object_detection_demo.gif)

*Real-time object detection running on Jetson Orin Nano*

**First time - create and run with a name:**

```bash
sudo docker run -it --name ultralytics-jetson \
  --ipc=host \
  --runtime=nvidia \
  --privileged \
  -v /ssd:/ssd \
  -w /ssd \
  -p 5000:5000 \
  ultralytics/ultralytics:latest-jetson-jetpack6
```

**Subsequent runs - just start the existing container:**

```bash
sudo docker start -ai ultralytics-jetson
```

**Install Flask for web-based applications:**

```bash
pip install flask
```

**Running Object Detection Web Interface:**

The Flask-based applications provide a web interface accessible from any browser on your network.

1. **Start the web server** (inside the Docker container):

```bash
# For real-time camera detection with web streaming
python src/simple_object_detection.py

# Or for the detection API server
python src/detection_server.py
```

2. **Access in your browser:**
```
http://<JETSON_IP>:5000
```

Replace `<JETSON_IP>` with your Jetson's IP address (e.g., `http://192.168.1.100:5000`)

**Example applications:**
- `src/simple_object_detection.py` - Real-time object detection with web streaming (Flask, port 5000)
- `src/detection_server.py` - Detection API server with web interface (Flask, port 5000)
- `src/video_detector.py` - Video file processing (command-line only)

### 2. LLM Inference (Local Language Models)

Run large language models locally on the Jetson Orin Nano using Ollama.

#### Option A: Native Installation (Recommended)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Start Ollama service:**

```bash
ollama serve
```

**Run a model (in a new terminal):**

```bash
# Lightweight models for 8GB Orin Nano
ollama run phi3
ollama run llama3.2:3b
ollama run qwen2.5:3b

# Larger models (may be slower)
ollama run llama3.2:8b
```

#### Option B: Docker Installation

**Using jetson-containers:**

```bash
jetson-containers run --name ollama \
  -v /ssd/ollama:/ollama \
  -e OLLAMA_MODELS=/ollama \
  $(autotag ollama)
```

**Access Ollama API:**

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "phi3",
  "prompt": "Why is the sky blue?"
}'
```

#### Optional: Open WebUI (Chat Interface)

Provides a ChatGPT-like interface for Ollama:

```bash
sudo docker run -d --network=host \
  -v /ssd/open-webui:/app/backend/data \
  -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

Access at `http://<JETSON_IP>:8080`

**Storage Requirements:**
- Ollama container: ~7GB
- Model sizes: 2GB-8GB per model
- Recommended: NVMe SSD with 64GB+ free space

### 3. Self-Driving Applications

Development environment for autonomous vehicle algorithms using the vision processing stack.

**Demo:**

![Street Object Detection](resources/street_object_detection.gif)

*YOLO11n detecting vehicles, pedestrians, traffic lights, and stop signs on street footage*

#### Model Setup and Optimization

This project uses YOLO11n optimized for the Jetson Orin Nano's GPU through TensorRT quantization:

**1. Download the pretrained model:**

```python
from ultralytics import YOLO

# Download YOLO11n PyTorch model
model = YOLO("yolo11n.pt")
```

**2. Export to TensorRT for GPU acceleration:**

```python
# Export the model to TensorRT engine format
model.export(format="engine")  # creates 'yolo11n.engine'

# Load the optimized TensorRT model
trt_model = YOLO("yolo11n.engine")
```

The TensorRT engine (`yolo11n.engine`) provides significant performance improvements on Jetson hardware compared to the original PyTorch model.

**Model files in this repo:**
- `models/yolo11n.pt` - Original PyTorch model (5.6 MB)
- `models/yolo11n.onnx` - ONNX intermediate format (10.7 MB)
- `models/yolo11n.engine` - TensorRT optimized for Jetson (12.5 MB)

#### Processing Videos with Object Detection

Use `video_detector.py` to process video files with self-driving relevant object detection:

```bash
# Process a video file
python src/video_detector.py <input_video.mp4> <output_video.mp4>

# Example
python src/video_detector.py street_footage.mp4 detected_street.mp4
```

**Detected object classes:**
- Vehicles: car, bus, truck, motorcycle, bicycle
- Pedestrians: person
- Traffic infrastructure: traffic light, stop sign

The detector filters YOLO's 80 classes to focus only on objects relevant for self-driving scenarios.

## Getting Started

(Add your specific instructions here)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
