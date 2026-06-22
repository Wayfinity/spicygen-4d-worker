# Use the official NVIDIA CUDA 11.8 base image
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

# 2. Clone 4C4D
WORKDIR /workspace
RUN git clone https://github.com/yangzf-1023/4C4D.git /workspace/4C4D

# Install local Python extension required by 4C4D train.py (fused_ssim import)
WORKDIR /workspace/4C4D
RUN pip install --no-cache-dir ./fused-ssim-main

# 3. Manually clone submodules to guarantee they are populated
# diff-gaussian-rasterization is on GitHub, simple-knn is on Inria GitLab
WORKDIR /workspace/4C4D/submodules
RUN rm -rf diff-gaussian-rasterization simple-knn && \
    git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization && \
    git clone https://gitlab.inria.fr/bkerbl/simple-knn.git

# 4. Build 4C4D Submodules with explicit GPU architecture flags
WORKDIR /workspace/4C4D/submodules/diff-gaussian-rasterization
# 8.0 is for A100, 8.6 is for 30-series, 8.9 is for 40-series, 9.0 is for H100
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
RUN pip install .

WORKDIR /workspace/4C4D/submodules/simple-knn
RUN pip install .

# Setup the RunPod Serverless Handler
WORKDIR /workspace
COPY rp_handler.py .

# Boot the RunPod listener
CMD ["python", "-u", "rp_handler.py"]