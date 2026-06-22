# Start from the official PyTorch image (Torch + CUDA 12.1 + NVCC already baked in)
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

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

# Install Serverless Requirements (No pip install torch step needed!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and setup MAtCha
RUN git clone https://github.com/anttwo/MAtCha.git /workspace/MAtCha
WORKDIR /workspace/MAtCha
RUN python install.py

# Clone and build 4C4D Submodules
WORKDIR /workspace
RUN git clone https://github.com/yangzf-1023/4C4D.git /workspace/4C4D

WORKDIR /workspace/4C4D/submodules/diff-gaussian-rasterization
RUN pip install .

WORKDIR /workspace/4C4D/submodules/simple-knn
RUN pip install .

# Setup the RunPod Serverless Handler
WORKDIR /workspace
COPY rp_handler.py .

# Boot the RunPod listener
CMD ["python", "-u", "rp_handler.py"]