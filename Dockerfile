# Use the official NVIDIA CUDA base image - This is guaranteed to exist
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Fix python symlinks
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# Install PyTorch + CUDA 11.8 specifically
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install Serverless Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 1. Clone and setup MAtCha 
RUN git clone https://github.com/anttwo/MAtCha.git /workspace/MAtCha
WORKDIR /workspace/MAtCha
RUN python install.py

# 2. Clone 4C4D WITH recursive submodules
WORKDIR /workspace
RUN git clone --recursive https://github.com/yangzf-1023/4C4D.git /workspace/4C4D

# 3. Build 4C4D Submodules
WORKDIR /workspace/4C4D/submodules/diff-gaussian-rasterization
RUN pip install .

WORKDIR /workspace/4C4D/submodules/simple-knn
RUN pip install .

# Setup the RunPod Serverless Handler
WORKDIR /workspace
COPY rp_handler.py .

CMD ["python", "-u", "rp_handler.py"]