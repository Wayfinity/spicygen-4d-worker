# SpicyGen 4D Gaussian Splatting Worker

Production-ready RunPod serverless worker for 4D Gaussian Splatting reconstruction from multi-view video.

## Pipeline

1. **Input**: 2x2 grid video (4 views)
2. **MAtCha SfM**: Structure-from-Motion to generate sparse point cloud
3. **4C4D Training**: 4D Gaussian Splatting optimization (1500 iterations)
4. **Output**: Binary `.splat` file + original video + 4 view crops → uploaded to S3

## Production Deployment

### Docker Build

```bash
docker build -t your-registry/spicygen-4d-worker:latest .
docker push your-registry/spicygen-4d-worker:latest
```

### RunPod Serverless Configuration

**GPU Requirement**: H100 (sm_90 architecture)

**Storage Volume**: Mount a persistent volume with the following structure:

```
/workspace/mast3r/checkpoints/
  ├── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth (2.6 GB)
  ├── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth (8.1 MB)
  └── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl (257 MB)

/workspace/Depth-Anything-V2/checkpoints/
  └── depth_anything_v2_vitl.pth (1.2 GB)
```

**Total storage needed**: ~4 GB for checkpoints + working space for job outputs

**Environment Variables** (set in RunPod serverless config):

```bash
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
AWS_BUCKET_NAME=your-bucket-name
```

### Downloading Checkpoints

```bash
# Create directories on storage volume
mkdir -p /path/to/volume/mast3r/checkpoints
mkdir -p /path/to/volume/Depth-Anything-V2/checkpoints

# Download MAtCha checkpoints
wget -P /path/to/volume/mast3r/checkpoints/ \
  https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

wget -P /path/to/volume/mast3r/checkpoints/ \
  https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth

wget -P /path/to/volume/mast3r/checkpoints/ \
  https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl

# Download Depth-Anything-V2 checkpoint
wget -P /path/to/volume/Depth-Anything-V2/checkpoints/ \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

### RunPod Volume Mount Points

When creating the serverless endpoint, mount the volume to these paths:
- `/workspace/mast3r` → your volume's `mast3r` directory
- `/workspace/Depth-Anything-V2` → your volume's `Depth-Anything-V2` directory

The handler will automatically symlink these to `/workspace/MAtCha/mast3r/checkpoints` and `/workspace/MAtCha/Depth-Anything-V2/checkpoints` on job start.

## Job Input Format

```json
{
  "input": {
    "video_url": "https://example.com/input_grid.mp4",
    "user_id": "user123",
    "job_id": "job456"
  }
}
```

## Job Output Format

**Success**:
```json
{
  "status": "COMPLETED",
  "spicygen_job_id": "job456",
  "s3_folder_path": "renders/user123/job456/",
  "splat_url": "https://presigned-url-for-scene_model_4d.splat",
  "files": ["input_grid.mp4", "view_0.mp4", "view_1.mp4", "view_2.mp4", "view_3.mp4", "scene_model_4d.splat"]
}
```

**Error**:
```json
{
  "error": "Pipeline error: [detailed error message]"
}
```

## Key Dependencies

- **PyTorch**: 2.0.1 + CUDA 11.8
- **MAtCha**: Latest from GitHub (cloned at build time)
- **4C4D**: Latest from GitHub (cloned at build time)
- **faiss-gpu-cu11**: 1.10.0
- **pytorch3d**: 0.7.4 (pre-built wheel)

## Upstream Patches

The Dockerfile applies patches to fix issues in the upstream MAtCha and 4C4D repositories:

### MAtCha Patches (`scripts/patch_matcha.py`)

**Issue**: `run_mast3r.py` uses `from_pretrained()` to load models, which goes through huggingface_hub's repo ID validation and fails with local `.pth` file paths.

**Fix**: Injects a `load_model_from_pth()` function that loads models directly from local checkpoint files, bypassing huggingface_hub validation. The patched code checks if the weights path is a local `.pth` file and uses the direct loader instead of `from_pretrained()`.

### 4C4D Patches (inline in Dockerfile)

**Issue 1**: `scene/dataset_readers.py` has image loading disabled (`image = None`) in two places for "lazy loading" performance, but the lazy loading was never implemented.

**Fix**: Uncomment `image = load_image(image_path)` and `temp_image = load_image(task['temp_path'])` to enable actual image loading.

**Issue 2**: The repository includes a custom `diff-gaussian-rasterization` with `sh_degree_t` support for time-varying spherical harmonics, but the Dockerfile was installing the generic version from graphdeco-inria which overwrites it.

**Fix**: Use 4C4D's custom rasterizer from `/workspace/4C4D/diff-gaussian-rasterization` instead of cloning the generic one.

## Build Optimizations

- H100-only (sm_90) for faster compilation
- Parallel CUDA builds (MAX_JOBS=4)
- Shallow git clones (--depth 1)
- Layer consolidation to reduce Docker cache invalidation
- pip cache mounting for faster rebuilds

## Testing Locally

To test the handler without building the full Docker image:

```bash
# Install dependencies
pip install -r requirements.txt

# Set up MAtCha and 4C4D (see Dockerfile for full setup steps)
# This is complex due to CUDA extensions - easier to test in a RunPod pod

# Run handler
python rp_handler.py
```

For faster iteration, use a RunPod GPU pod with the base CUDA image and manually install dependencies as described in the Dockerfile.
