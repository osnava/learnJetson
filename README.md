# Self-Driving Algorithm - Jetson Orin Nano Deployment

This repository contains a self-driving algorithm designed to run on the NVIDIA Jetson Orin Nano.

## Prerequisites

- NVIDIA Jetson Orin Nano with JetPack 6
- Docker with NVIDIA runtime support
- SSD mounted at `/ssd`

## Setup

### Build Docker Image

Build the custom Docker image with nano and additional tools:

```bash
sudo docker build -t selfdriving-jetson .
```

### Run Container

To create and run the container:

```bash
sudo docker run -it \
  --ipc=host \
  --runtime=nvidia \
  --privileged \
  -v /ssd:/ssd \
  -w /ssd \
  -p 5000:5000 \
  selfdriving-jetson
```

**Container Configuration:**
- `--ipc=host`: Enables shared memory access for efficient data processing
- `--runtime=nvidia`: Enables GPU access within the container
- `--privileged`: Grants extended privileges for hardware access
- `-v /ssd:/ssd`: Mounts the SSD storage into the container
- `-w /ssd`: Sets the working directory to the SSD mount point
- `-p 5000:5000`: Exposes port 5000 for web interfaces or APIs

## Getting Started

(Add your specific instructions here)

## License

(Add license information)
