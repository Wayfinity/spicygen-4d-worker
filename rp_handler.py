import os
import shutil
import subprocess
import urllib.request
import glob
import runpod
import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError
from plyfile import PlyData

# --- Workspace Configuration ---
WORKSPACE = "/workspace"
INPUT_DIR = os.path.join(WORKSPACE, "inputs")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")


def run_cmd(cmd, cwd=None):
    """Run a shell command and return the result."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


def extract_frames(video_path, output_dir, fps=4):
    """Extract frames from video at specified FPS."""
    os.makedirs(output_dir, exist_ok=True)
    frame_pattern = os.path.join(output_dir, "frame_%04d.png")

    run_cmd([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        frame_pattern
    ])

    frames = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    return frames


def convert_ply_to_splat(ply_path, splat_path):
    """Convert a .ply file to .splat format."""
    print(f"Converting {ply_path} to {splat_path}")

    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    num_points = len(vertex.data)
    print(f"Processing {num_points} points")
    
    with open(splat_path, 'wb') as f:
        for i in range(num_points):
            v = vertex.data[i]

            # Position (x, y, z)
            x, y, z = v['x'], v['y'], v['z']
            f.write(np.array([x, y, z], dtype=np.float32).tobytes())

            # Scale (scale_0, scale_1, scale_2) - already in log space
            if 'scale_0' in vertex.dtype.names:
                s0, s1, s2 = v['scale_0'], v['scale_1'], v['scale_2']
                f.write(np.array([s0, s1, s2], dtype=np.float32).tobytes())
            else:
                # Default scale if not present
                f.write(np.array([-10.0, -10.0, -10.0],
                        dtype=np.float32).tobytes())

            # Color (r, g, b) - convert from SH coefficients if needed
            if 'f_dc_0' in vertex.dtype.names:
                # Convert from SH to RGB (simplified)
                r = max(0, min(255, int((v['f_dc_0'] * 0.282 + 0.5) * 255)))
                g = max(0, min(255, int((v['f_dc_1'] * 0.282 + 0.5) * 255)))
                b = max(0, min(255, int((v['f_dc_2'] * 0.282 + 0.5) * 255)))
            elif 'red' in vertex.dtype.names:
                r, g, b = v['red'], v['green'], v['blue']
            else:
                r, g, b = 128, 128, 128

            # Opacity
            if 'opacity' in vertex.dtype.names:
                opacity = max(0, min(255, int(v['opacity'] * 255)))
            else:
                opacity = 255

            f.write(bytes([r, g, b, opacity]))

            # Rotation quaternion (rot_0, rot_1, rot_2, rot_3)
            if 'rot_0' in vertex.dtype.names:
                q0, q1, q2, q3 = v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']
                # Normalize
                norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
                if norm > 0:
                    q0, q1, q2, q3 = q0/norm, q1/norm, q2/norm, q3/norm
                # Convert to byte format (128 = identity)
                f.write(bytes([
                    int((q0 + 1.0) * 127.5),
                    int((q1 + 1.0) * 127.5),
                    int((q2 + 1.0) * 127.5),
                    int((q3 + 1.0) * 127.5)
                ]))
            else:
                # Identity quaternion
                f.write(bytes([128, 128, 128, 128]))

    print(f"Written {splat_path}")


def find_latest_ply(model_path):
    """Find the latest .ply file in the model directory."""
    ply_files = glob.glob(os.path.join(
        model_path, "point_cloud", "iteration_*", "point_cloud.ply"))
    if not ply_files:
        raise RuntimeError("No .ply files found in model directory")
    return max(ply_files, key=os.path.getmtime)


def cleanup_workspace():
    """Clean up the workspace directories."""
    for dir_path in [INPUT_DIR, OUTPUT_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)


def handler(job):
    """Main handler function for processing video to 3D Gaussian Splat."""
    job_input = job.get("input", {})

    video_url = job_input.get("video_url")
    user_id = job_input.get("user_id")
    job_id = job_input.get("job_id")

    if not video_url or not user_id or not job_id:
        return {"error": "Missing required payload data. Must include video_url, user_id, and job_id."}

    cleanup_workspace()

    video_path = os.path.join(INPUT_DIR, "input_video.mp4")
    splat_output = os.path.join(OUTPUT_DIR, "scene_model_3d.splat")

    try:
        print(f"[{job_id}] Downloading source video...")
        urllib.request.urlretrieve(video_url, video_path)

        # Extract frames at 4 FPS
        print(f"[{job_id}] Extracting frames from orbiting video...")
        frames_dir = os.path.join(INPUT_DIR, "frames")
        frames = extract_frames(video_path, frames_dir, fps=4)
        print(f"[{job_id}] Extracted {len(frames)} frames")

        if len(frames) < 10:
            return {"error": f"Too few frames extracted ({len(frames)}). Need at least 10 for reliable reconstruction."}

        # Run COLMAP for camera pose estimation
        print(f"[{job_id}] Running COLMAP SfM...")
        colmap_dir = os.path.join(OUTPUT_DIR, "colmap")
        os.makedirs(colmap_dir, exist_ok=True)

        db_path = os.path.join(colmap_dir, "database.db")
        sparse_dir = os.path.join(colmap_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)

        # Feature extraction
        run_cmd([
            "colmap", "feature_extractor",
            "--database_path", db_path,
            "--image_path", frames_dir,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE",
            "--SiftExtraction.use_gpu", "1",
        ])

        # Feature matching
        run_cmd([
            "colmap", "exhaustive_matcher",
            "--database_path", db_path,
            "--SiftMatching.use_gpu", "1",
        ])

        # 3D reconstruction
        run_cmd([
            "colmap", "mapper",
            "--database_path", db_path,
            "--image_path", frames_dir,
            "--output_path", sparse_dir,
        ])

        # Check if COLMAP succeeded
        colmap_sparse_dir = os.path.join(sparse_dir, "0")
        if not os.path.exists(os.path.join(colmap_sparse_dir, "images.bin")):
            return {"error": "COLMAP failed to reconstruct the scene. The video may not have enough visual features or camera motion."}

        # Prepare 3DGS input structure (standard COLMAP format)
        print(f"[{job_id}] Preparing 3DGS input...")
        gs_source = os.path.join(OUTPUT_DIR, "gaussian_splatting_source")
        gs_images = os.path.join(gs_source, "images")
        gs_sparse = os.path.join(gs_source, "sparse")
        os.makedirs(gs_images, exist_ok=True)
        os.makedirs(gs_sparse, exist_ok=True)

        # Copy images
        for frame in frames:
            shutil.copy2(frame, os.path.join(
                gs_images, os.path.basename(frame)))

        # Copy COLMAP sparse files
        shutil.copytree(colmap_sparse_dir, os.path.join(
            gs_sparse, "0"), dirs_exist_ok=True)

        # Convert COLMAP .bin to .txt for 3DGS compatibility
        run_cmd([
            "colmap", "model_converter",
            "--input_path", os.path.join(gs_sparse, "0"),
            "--output_path", os.path.join(gs_sparse, "0"),
            "--output_type", "TXT",
        ])

        # Train 3D Gaussian Splatting
        print(f"[{job_id}] Training 3D Gaussian Splatting (3000 iterations)...")
        gs_model_path = os.path.join(OUTPUT_DIR, "gaussian_splatting_model")
        os.makedirs(gs_model_path, exist_ok=True)

        run_cmd([
            "python3", "/workspace/4C4D/train.py",
            "-s", gs_source,
            "-m", gs_model_path,
            "--iterations", "3000",
            "--test_iterations", "-1",
            "--save_iterations", "3000",
            "--checkpoint_iterations", "3000",
        ], cwd="/workspace/4C4D")

        # Convert PLY to SPLAT
        print(f"[{job_id}] Converting to .splat format...")
        ply_path = find_latest_ply(gs_model_path)
        convert_ply_to_splat(ply_path, splat_output)

        print(f"[{job_id}] Success! Generated 3D Gaussian Splat.")
        return {
            "status": "success",
            "splat_path": splat_output,
            "num_frames": len(frames),
            "message": "3D Gaussian Splatting completed successfully. You can view the .splat file in any compatible viewer."
        }

    except Exception as e:
        print(f"[{job_id}] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
