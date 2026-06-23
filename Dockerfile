# ============================================================
# SpicyGen 4D Gaussian Splatting Worker — Production Image
# Optimized for H100 (sm_90)
# ============================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="9.0" \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=4

# ── System dependencies + python symlink ────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git ffmpeg libgl1-mesa-glx \
    libglib2.0-0 wget cmake libcgal-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# ── PyTorch 2.0.1 + CUDA 11.8 + faiss + pytorch3d ──────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir faiss-gpu-cu11==1.10.0 \
    && pip install --no-cache-dir pytorch3d==0.7.4 \
       -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html

# ── Serverless handler dependencies ─────────────────────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir runpod==1.7.0

# ════════════════════════════════════════════════════════════
# MAtCha
# ════════════════════════════════════════════════════════════
RUN git clone --depth 1 https://github.com/anttwo/MAtCha.git /workspace/MAtCha

# ── All MAtCha CUDA extensions in one layer ─────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-build-isolation /workspace/MAtCha/mast3r/dust3r/croco/models/curope \
    && cd /workspace/MAtCha/mast3r/asmk \
    && cythonize cython/*.pyx \
    && pip install --no-cache-dir . \
    && cd /workspace/MAtCha/2d-gaussian-splatting/submodules \
    && pip install --no-cache-dir ./diff-surfel-rasterization \
    && pip install --no-cache-dir ./simple-knn \
    && cd tetra-triangulation \
    && cmake -DCMAKE_CXX_FLAGS="-I${CUDA_HOME}/include" . \
    && make -j${MAX_JOBS} \
    && pip install --no-cache-dir .

# ════════════════════════════════════════════════════════════
# 4C4D
# ════════════════════════════════════════════════════════════
RUN git clone --depth 1 --recursive https://github.com/yangzf-1023/4C4D.git /workspace/4C4D

# ── All 4C4D extensions in one layer ────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-build-isolation /workspace/4C4D/fused-ssim-main \
    && mkdir -p /workspace/4C4D/submodules \
    && cd /workspace/4C4D/submodules \
    && rm -rf diff-gaussian-rasterization simple-knn \
    && git clone --depth 1 --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization \
    && git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git \
    && pip install --no-cache-dir ./diff-gaussian-rasterization \
    && pip install --no-cache-dir ./simple-knn \
    && pip install --no-cache-dir /workspace/4C4D/pointops2 \
    && sed -i 's/# image = load_image(image_path)/image = load_image(image_path)/' /workspace/4C4D/scene/dataset_readers.py \
    && sed -i '/^[[:space:]]*image = None$/d' /workspace/4C4D/scene/dataset_readers.py

# ════════════════════════════════════════════════════════════
# RunPod Serverless Handler
# ════════════════════════════════════════════════════════════
WORKDIR /workspace
COPY rp_handler.py .

CMD ["python", "-u", "rp_handler.py"]
