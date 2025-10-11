FROM ultralytics/ultralytics:latest-jetson-jetpack6

# Install nano and other useful tools
RUN apt-get update && apt-get install -y \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ssd
