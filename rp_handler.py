import os
import shutil
import subprocess
import struct
import urllib.request
import glob
import sqlite3
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
    print(f"[cmd] {' '.join(cmd)}")
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd, output=completed.stdout)


def build_4c4d_config(config_path: str, source_path: str, model_path: str, iterations: int = 1500):
    # 4C4D now expects a YAML config passed via --config.
    config_content = f"""
gaussian_dim: 4
time_duration: [0.0, 10.0]
num_pts: 200000
num_pts_ratio: 1.0
rot_4d: true
force_sh_3d: false
batch_size: 1
exhaust_test: false

ModelParams:
  sh_degree: 3
  source_path: "{source_path}"
  model_path: "{model_path}"
  images: "images"
  resolution: 1
  white_background: false
  data_device: "cuda"
  eval: false
  extension: ".png"
  num_extra_pts: 0
  loaded_pth: ""
  frame_ratio: 1
  dataloader: false

PipelineParams:
  convert_SHs_python: false
  compute_cov3D_python: false
  debug: false
  env_map_res: 0
  env_optimize_until: 1000000000
  env_optimize_from: 0
  eval_shfs_4d: true

OptimizationParams:
  iterations: {iterations}
""".strip()

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)


def find_latest_4c4d_ply(model_path: str):
    pattern = os.path.join(model_path, "point_cloud", "iteration_*", "point_cloud.ply")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No 4C4D PLY found using pattern: {pattern}")

    def iteration_key(path):
        folder = os.path.basename(os.path.dirname(path))
        try:
            return int(folder.split("_")[-1])
        except ValueError:
            return -1

    return max(candidates, key=iteration_key)


def cleanup_workspace():
    for path in [INPUT_DIR, OUTPUT_DIR]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

def convert_ply_to_splat(ply_input_path: str, splat_output_path: str):
    if not os.path.exists(ply_input_path):
        raise FileNotFoundError(f"Source PLY file not found at {ply_input_path}")
        
    plydata = PlyData.read(ply_input_path)
    vertex = plydata['vertex']

    x = np.asarray(vertex['x'], dtype=np.float32)
    y = np.asarray(vertex['y'], dtype=np.float32)
    z = np.asarray(vertex['z'], dtype=np.float32)
    
    scale_0 = np.asarray(vertex['scale_0'], dtype=np.float32)
    scale_1 = np.asarray(vertex['scale_1'], dtype=np.float32)
    scale_2 = np.asarray(vertex['scale_2'], dtype=np.float32)

    r = np.asarray(vertex['f_dc_0'], dtype=np.float32) if 'f_dc_0' in vertex else np.asarray(vertex['red'], dtype=np.float32)
    g = np.asarray(vertex['f_dc_1'], dtype=np.float32) if 'f_dc_1' in vertex else np.asarray(vertex['green'], dtype=np.float32)
    b = np.asarray(vertex['f_dc_2'], dtype=np.float32) if 'f_dc_2' in vertex else np.asarray(vertex['blue'], dtype=np.float32)
    
    opacity = np.asarray(vertex['opacity'], dtype=np.float32)
    
    rot_0 = np.asarray(vertex['rot_0'], dtype=np.float32)
    rot_1 = np.asarray(vertex['rot_1'], dtype=np.float32)
    rot_2 = np.asarray(vertex['rot_2'], dtype=np.float32)
    rot_3 = np.asarray(vertex['rot_3'], dtype=np.float32)
    
    num_primitives = len(x)

    with open(splat_output_path, 'wb') as f:
        for i in range(num_primitives):
            res_r = int(np.clip(r[i] * 255, 0, 255))
            res_g = int(np.clip(g[i] * 255, 0, 255))
            res_b = int(np.clip(b[i] * 255, 0, 255))
            res_a = int(np.clip(1.0 / (1.0 + np.exp(-opacity[i])) * 255, 0, 255)) 

            f.write(struct.pack('fff', x[i], y[i], z[i]))
            f.write(struct.pack('fff', np.exp(scale_0[i]), np.exp(scale_1[i]), np.exp(scale_2[i])))
            f.write(struct.pack('BBBB', res_r, res_g, res_b, res_a))

            q = np.array([rot_0[i], rot_1[i], rot_2[i], rot_3[i]])
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
            f.write(struct.pack('BBBB', int((q[0]+1)*127.5), int((q[1]+1)*127.5), int((q[2]+1)*127.5), int((q[3]+1)*127.5)))


def extract_frames(video_path: str, output_dir: str, fps: int = 4):
    """Extract frames from video at specified FPS."""
    os.makedirs(output_dir, exist_ok=True)
    run_cmd([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(output_dir, "frame_%04d.png"),
    ])
    return sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))


def run_colmap(images_dir: str, output_dir: str):
    """Run COLMAP SfM pipeline to estimate camera poses."""
    os.makedirs(output_dir, exist_ok=True)

    # Feature extraction
    run_cmd([
        "colmap", "feature_extractor",
        "--database_path", os.path.join(output_dir, "database.db"),
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE",
        "--SiftExtraction.use_gpu", "1",
    ])

    # Feature matching
    run_cmd([
        "colmap", "exhaustive_matcher",
        "--database_path", os.path.join(output_dir, "database.db"),
        "--SiftMatching.use_gpu", "1",
    ])

    # Reconstruction
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    run_cmd([
        "colmap", "mapper",
        "--database_path", os.path.join(output_dir, "database.db"),
        "--image_path", images_dir,
        "--output_path", os.path.join(output_dir, "sparse"),
    ])

    return sparse_dir


def handler(job):
    job_input = job.get("input", {})

    video_url = job_input.get("video_url")
    user_id = job_input.get("user_id")
    job_id = job_input.get("job_id")

    if not video_url or not user_id or not job_id:
        return {"error": "Missing required payload data. Must include video_url, user_id, and job_id."}

    cleanup_workspace()

    video_path = os.path.join(INPUT_DIR, "input_video.mp4")
    splat_output = os.path.join(OUTPUT_DIR, "scene_model_4d.splat")

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
        sparse_dir = run_colmap(frames_dir, colmap_dir)

        # Prepare 4C4D input structure
        print(f"[{job_id}] Preparing 4C4D input...")
        c4d_source = os.path.join(OUTPUT_DIR, "4c4d_source")
        c4d_images = os.path.join(c4d_source, "images")
        os.makedirs(c4d_images, exist_ok=True)

        # Copy COLMAP sparse reconstruction (excluding any images/ subdir)
        dst_sparse = os.path.join(c4d_source, "sparse", "0")
        os.makedirs(dst_sparse, exist_ok=True)
        for item in os.listdir(sparse_dir):
            src = os.path.join(sparse_dir, item)
            dst = os.path.join(dst_sparse, item)
            if os.path.isdir(src) and item == "images":
                continue  # Skip images subdir if present
            shutil.copy2(src, dst) if os.path.isfile(
                src) else shutil.copytree(src, dst)

        # Copy only the frames that COLMAP successfully reconstructed
        # Read image names directly from COLMAP's images.bin
        import sys
        sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages")
        try:
            from colmap import read_write_model
            cam_extrinsics = read_write_model.read_images_binary(
                os.path.join(dst_sparse, "images.bin"))
            colmap_image_names = set(
                img.name for img in cam_extrinsics.values())
        except Exception as e:
            print(f"[{job_id}] Warning: Could not read images.bin: {e}")
            colmap_image_names = set(os.path.basename(f) for f in frames)

        print(f"[{job_id}] COLMAP reconstructed {len(colmap_image_names)} images")

        for frame in frames:
            frame_name = os.path.basename(frame)
            if frame_name in colmap_image_names:
                shutil.copy2(frame, os.path.join(c4d_images, frame_name))
            else:
                print(
                    f"[{job_id}] Skipping {frame_name} (not in COLMAP reconstruction)")

        # Run 4C4D training
        print(f"[{job_id}] Running 4C4D optimization (1500 iterations)...")
        c4d_model_path = os.path.join(OUTPUT_DIR, "4c4d_model")
        c4d_config_path = os.path.join(OUTPUT_DIR, "4c4d_config.yaml")
        build_4c4d_config(
            config_path=c4d_config_path,
            source_path=c4d_source,
            model_path=c4d_model_path,
            iterations=1500,
        )
        run_cmd([
            "python3", "/workspace/4C4D/train.py",
            "--config", c4d_config_path,
            "--save_iterations", "1500",
        ], cwd="/workspace/4C4D")

        # Convert PLY to SPLAT
        print(f"[{job_id}] Converting PLY to binary SPLAT format...")
        ply_input = find_latest_4c4d_ply(c4d_model_path)
        convert_ply_to_splat(ply_input, splat_output)

        # Upload to S3
        print(f"[{job_id}] Uploading to AWS S3...")
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )

        bucket_name = os.environ.get(
            "AWS_BUCKET_NAME", "your-production-bucket-name")
        base_s3_folder = f"renders/{user_id}/{job_id}"

        files_to_upload = [
            ("input_video.mp4", video_path),
            ("scene_model_4d.splat", splat_output)
        ]

        uploaded_files = []
        presigned_splat_url = None

        for file_name, local_path in files_to_upload:
            if os.path.exists(local_path):
                s3_key = f"{base_s3_folder}/{file_name}"
                s3_client.upload_file(local_path, bucket_name, s3_key)
                uploaded_files.append(file_name)

                if file_name == "scene_model_4d.splat":
                    presigned_splat_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': bucket_name, 'Key': s3_key},
                        ExpiresIn=3600
                    )

        print(f"[{job_id}] Process complete.")
        return {
            "status": "COMPLETED",
            "spicygen_job_id": job_id,
            "s3_folder_path": f"{base_s3_folder}/",
            "splat_url": presigned_splat_url,
            "files": uploaded_files
        }

    except subprocess.CalledProcessError as e:
        error_msg = e.output or e.stderr or e.stdout or "Subprocess execution failed"
        return {"error": f"Pipeline error: {error_msg}"}
    except NoCredentialsError:
        return {"error": "AWS credentials not available in RunPod environment secrets."}
    except Exception as e:
        return {"error": f"Internal Process Failure: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})