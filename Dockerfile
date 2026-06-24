# ============================================================
# SpicyGen 4D Gaussian Splatting Worker — Production Image
# Optimized for H100 (sm_90)
# ============================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="9.0" \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4

# ── System dependencies + python symlink ───────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git ffmpeg libgl1-mesa-glx \
    libglib2.0-0 wget cmake ninja-build gcc g++ \
    libcgal-dev libeigen3-dev \
    libboost-program-options-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-system-dev libboost-test-dev \
    libflann-dev libfreeimage-dev libgflags-dev libglew-dev \
    libglfw3-dev libgoogle-glog-dev libmetis-dev libsqlite3-dev \
    libceres-dev libsuitesparse-dev libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ── COLMAP from source ─────────────────────────────────────
RUN git clone https://github.com/colmap/colmap.git /tmp/colmap \
    && cd /tmp/colmap \
    && git checkout 3.9.1 \
    && mkdir build && cd build \
    && cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=90 -DGUI_ENABLED=OFF \
    && ninja -j${MAX_JOBS} \
    && ninja install \
    && rm -rf /tmp/colmap

WORKDIR /workspace

# ── PyTorch 2.0.1 + CUDA 11.8 + pytorch3d + faiss ──────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir pytorch3d==0.7.4 \
       -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html \
    && pip install --no-cache-dir faiss-cpu==1.10.0

# ── Serverless handler dependencies ─────────────────────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir runpod==1.7.0

# ════════════════════════════════════════════════════════════
# 4C4D
# ════════════════════════════════════════════════════════════
RUN git clone --depth 1 --recursive https://github.com/yangzf-1023/4C4D.git /workspace/4C4D

# ── All 4C4D extensions in one layer ────────────────────────
# fused-ssim uses CUDA_ARCHITECTURES env var (not TORCH_CUDA_ARCH_LIST)
RUN --mount=type=cache,target=/root/.cache/pip \
    CUDA_ARCHITECTURES="90" pip install --no-cache-dir --no-build-isolation /workspace/4C4D/fused-ssim-main \
    && pip install --no-cache-dir /workspace/4C4D/diff-gaussian-rasterization \
    && mkdir -p /workspace/4C4D/submodules \
    && cd /workspace/4C4D/submodules \
    && git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git \
    && pip install --no-cache-dir ./simple-knn \
    && pip install --no-cache-dir /workspace/4C4D/pointops2
COPY scripts/patch_4c4d.py /tmp/patch_4c4d.py
RUN python3 /tmp/patch_4c4d.py && \
    echo "=== Verifying 4C4D patch ===" && \
    grep -n "image = load_image" /workspace/4C4D/scene/dataset_readers.py | head -2 && \
    echo "=== 4C4D patch verified ===" && \
    rm /tmp/patch_4c4d.py

# ── Patch 4C4D train.py for PyTorch 2.0.1 compatibility ─────
COPY scripts/patch_4c4d_train.py /tmp/patch_4c4d_train.py
RUN python3 /tmp/patch_4c4d_train.py && rm /tmp/patch_4c4d_train.py

# ════════════════════════════════════════════════════════════
# RunPod Serverless Handler
# ════════════════════════════════════════════════════════════
WORKDIR /workspace
COPY rp_handler.py .

CMD ["python", "-u", "rp_handler.py"]
