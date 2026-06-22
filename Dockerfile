# ============================================================
# SpicyGen 4D Gaussian Splatting Worker — Production Image
# ============================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Fix python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# ── GPU architecture targets (set ONCE, used by all CUDA extension builds) ──
# 8.0=A100  8.6=3090/A6000  8.9=L40/4090  9.0=H100
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# ── PyTorch 2.0.1 + CUDA 11.8 ──────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ── GPU-specific packages (must match CUDA 11.8) ───────────
RUN pip install --no-cache-dir faiss-gpu-cu11==1.10.0

# ── pytorch3d 0.7.4 (MAtCha 3D transforms) ─────────────────
# Pre-built wheel: Python 3.10 + CUDA 11.8 + PyTorch 2.0.1
RUN pip install --no-cache-dir pytorch3d==0.7.4 \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html

# ── Serverless handler dependencies ─────────────────────────
# Split into two steps: gradio==4.44.1 pins tomlkit==0.12.0 while
# runpod==1.7.0 needs tomlkit>=0.12.2 — they can't co-resolve.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod==1.7.0

# ════════════════════════════════════════════════════════════
# MAtCha
# ════════════════════════════════════════════════════════════
RUN git clone https://github.com/anttwo/MAtCha.git /workspace/MAtCha

# ── RoPE2D CUDA kernels ────────────────────────────────────
# MAtCha falls back to slow PyTorch ops if these are not compiled.
WORKDIR /workspace/MAtCha/mast3r/dust3r/croco/models/curope
RUN pip install --no-cache-dir --no-build-isolation .

# ── ASMK retrieval module ───────────────────────────────────
# Cython files must be pre-compiled before setup.py runs.
WORKDIR /workspace/MAtCha/mast3r/asmk
RUN cythonize cython/*.pyx && pip install --no-cache-dir .

# ── MAtCha 2D Gaussian Splatting CUDA extensions ────────────
WORKDIR /workspace/MAtCha/2d-gaussian-splatting/submodules/diff-surfel-rasterization
RUN pip install --no-cache-dir .

WORKDIR /workspace/MAtCha/2d-gaussian-splatting/submodules/simple-knn
RUN pip install --no-cache-dir .

WORKDIR /workspace/MAtCha/2d-gaussian-splatting/submodules/tetra-triangulation
RUN cmake . && make && pip install --no-cache-dir .

# ════════════════════════════════════════════════════════════
# 4C4D
# ════════════════════════════════════════════════════════════
WORKDIR /workspace
RUN git clone https://github.com/yangzf-1023/4C4D.git /workspace/4C4D

# ── fused_ssim (SSIM loss used by 4C4D train.py) ───────────
WORKDIR /workspace/4C4D
RUN pip install --no-cache-dir --no-build-isolation ./fused-ssim-main

# ── 4C4D GPU submodules ─────────────────────────────────────
WORKDIR /workspace/4C4D/submodules
RUN rm -rf diff-gaussian-rasterization simple-knn && \
    git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization && \
    git clone https://gitlab.inria.fr/bkerbl/simple-knn.git

WORKDIR /workspace/4C4D/submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir .

WORKDIR /workspace/4C4D/submodules/simple-knn
RUN pip install --no-cache-dir .

# ── pointops2 (CUDA ops used by 4C4D point cloud processing) ─
WORKDIR /workspace/4C4D/pointops2
RUN pip install --no-cache-dir .

# ════════════════════════════════════════════════════════════
# RunPod Serverless Handler
# ════════════════════════════════════════════════════════════
WORKDIR /workspace
COPY rp_handler.py .

CMD ["python", "-u", "rp_handler.py"]
