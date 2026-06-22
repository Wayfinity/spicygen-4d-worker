# Use a development image that contains the nvcc compiler
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1. Install PyTorch specifically for CUDA 12.1 BEFORE requirements
# This is mandatory for diff-gaussian-rasterization to compile
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install Serverless Requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 3. Clone and setup MAtCha
RUN git clone https://github.com/anttwo/MAtCha.git /workspace/MAtCha
WORKDIR /workspace/MAtCha
RUN python3 install.py

# 4. Clone and build 4C4D Submodules
WORKDIR /workspace
RUN git clone https://github.com/yangzf-1023/4C4D.git /workspace/4C4D

WORKDIR /workspace/4C4D/submodules/diff-gaussian-rasterization
RUN pip3 install .

WORKDIR /workspace/4C4D/submodules/simple-knn
RUN pip3 install .

# 5. Setup the RunPod Serverless Handler
WORKDIR /workspace
COPY rp_handler.py .

# Boot the RunPod listener (No EXPOSE port needed for serverless)
CMD ["python3", "-u", "rp_handler.py"]