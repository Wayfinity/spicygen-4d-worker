# Switch to CUDA 11.8 to perfectly match MAtCha's strict internal requirements
# Updated tag that actually exists on Docker Hub
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8

ENV DEBIAN_FRONTEND=noninteractive

# Install critical system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Serverless Requirements into the global base environment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 1. Clone and setup MAtCha 
# (This script will automatically create an isolated Conda env named "matcha")
RUN git clone https://github.com/anttwo/MAtCha.git /workspace/MAtCha
WORKDIR /workspace/MAtCha
RUN python install.py

# 2. Clone 4C4D WITH the --recursive flag to pull the C++ submodule folders!
WORKDIR /workspace
RUN git clone --recursive https://github.com/yangzf-1023/4C4D.git /workspace/4C4D

# 3. Build 4C4D Submodules natively in the base environment
WORKDIR /workspace/4C4D/submodules/diff-gaussian-rasterization
RUN pip install .

WORKDIR /workspace/4C4D/submodules/simple-knn
RUN pip install .

# Setup the RunPod Serverless Handler
WORKDIR /workspace
COPY rp_handler.py .

CMD ["python", "-u", "rp_handler.py"]