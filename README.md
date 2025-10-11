# Self-Driving Algorithm - Jetson Orin Nano Deployment

This repository contains a self-driving algorithm designed to run on the NVIDIA Jetson Orin Nano.

## Prerequisites

- NVIDIA Jetson Orin Nano with JetPack 6
- Docker with NVIDIA runtime support
- SSD mounted at `/ssd`

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
