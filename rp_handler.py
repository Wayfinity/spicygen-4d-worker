import os
import shutil
import subprocess
import struct
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


def find_matcha_points_file(output_root: str):
    preferred_paths = [
        os.path.join(output_root, "mast3r_sfm", "points.ply"),
        os.path.join(output_root, "mast3r_sfm", "point_cloud.ply"),
        os.path.join(output_root, "points.ply"),
        os.path.join(output_root, "point_cloud.ply"),
    ]
    for path in preferred_paths:
        if os.path.exists(path):
            return path

    candidates = glob.glob(os.path.join(output_root, "**", "points.ply"), recursive=True)
    candidates.extend(glob.glob(os.path.join(output_root, "**", "point_cloud.ply"), recursive=True))
    if candidates:
        return sorted(set(candidates))[0]

    raise FileNotFoundError(
        f"MAtCha did not produce a point cloud under {output_root}. "
        f"Checked: {', '.join(preferred_paths)}"
    )

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

def handler(job):
    job_input = job.get("input", {})

    video_url = job_input.get("video_url")
    user_id = job_input.get("user_id")
    job_id = job_input.get("job_id")

    if not video_url or not user_id or not job_id:
        return {"error": "Missing required payload data. Must include video_url, user_id, and job_id."}

    cleanup_workspace()
    
    grid_video_path = os.path.join(INPUT_DIR, "input_grid.mp4")
    splat_output = os.path.join(OUTPUT_DIR, "scene_model_4d.splat")

    try:
        print(f"[{job_id}] Downloading source video...")
        urllib.request.urlretrieve(video_url, grid_video_path)

        print(f"[{job_id}] Slicing grid via FFmpeg into agnostic indexed views...")
        run_cmd([
            "ffmpeg", "-y", "-i", grid_video_path,
            "-filter_complex",
            "[0:v]crop=iw/2:ih/2:0:0[tl];[0:v]crop=iw/2:ih/2:iw/2:0[tr];"
            "[0:v]crop=iw/2:ih/2:0:ih/2[bl];[0:v]crop=iw/2:ih/2:iw/2:ih/2[br]",
            "-map", "[tl]", os.path.join(INPUT_DIR, "view_0.mp4"),
            "-map", "[tr]", os.path.join(INPUT_DIR, "view_1.mp4"),
            "-map", "[bl]", os.path.join(INPUT_DIR, "view_2.mp4"),
            "-map", "[br]", os.path.join(INPUT_DIR, "view_3.mp4"),
        ])

        print(f"[{job_id}] Running MAtCha SfM initialization...")
        matcha_images_dir = os.path.join(OUTPUT_DIR, "init_points", "images")
        os.makedirs(matcha_images_dir, exist_ok=True)
        matcha_output_dir = os.path.join(OUTPUT_DIR, "init_points", "mast3r_sfm")

        for i in range(4):
            run_cmd([
                "ffmpeg", "-y", "-i", os.path.join(INPUT_DIR, f"view_{i}.mp4"),
                "-frames:v", "1",
                os.path.join(matcha_images_dir, f"{i:04d}.png"),
            ])

        run_cmd([
            "python3", "/workspace/MAtCha/mast3r/run_mast3r.py",
            "--scene_path", matcha_images_dir,
            "--output_dir", matcha_output_dir,
            "--weights_path", "/workspace/MAtCha/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            "--retrieval_model", "/workspace/MAtCha/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
            "--min_conf_thr", "0.0",
            "--matching_conf_thr", "0.0",
            "--n_coarse_iterations", "1000",
            "--n_refinement_iterations", "1000",
            "--TSDF_thresh", "0.0",
            "--fix_principal_point",
            "--n_images", "4",
            "--image_size", "512",
            "--max_window_size", "20",
            "--max_refid", "10",
            "--output_conf_thr", "0.1",
        ], cwd="/workspace/MAtCha")

        matcha_points = find_matcha_points_file(os.path.join(OUTPUT_DIR, "init_points"))

        # Keep legacy output path for any downstream steps expecting point_cloud.ply.
        shutil.copyfile(matcha_points, os.path.join(OUTPUT_DIR, "init_points", "point_cloud.ply"))

        # Restructure for 4C4D: it expects COLMAP at <source>/sparse/0/ and images at <source>/images/
        init_points_dir = os.path.join(OUTPUT_DIR, "init_points")
        c4d_source = os.path.join(init_points_dir, "4c4d_source")
        c4d_images = os.path.join(c4d_source, "images")
        os.makedirs(c4d_images, exist_ok=True)
        shutil.copytree(
            os.path.join(init_points_dir, "mast3r_sfm", "sparse"),
            os.path.join(c4d_source, "sparse"),
            dirs_exist_ok=True,
        )
        for img in os.listdir(os.path.join(init_points_dir, "images")):
            if img.endswith(".png"):
                shutil.copy2(
                    os.path.join(init_points_dir, "images", img),
                    os.path.join(c4d_images, img),
                )

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
        ], cwd="/workspace/4C4D")

        print(f"[{job_id}] Converting PLY to binary SPLAT format...")
        ply_input = find_latest_4c4d_ply(c4d_model_path)
        convert_ply_to_splat(ply_input, splat_output)
        
        # --- Batch S3 Upload Logic ---
        print(f"[{job_id}] Uploading all assets to AWS S3...")
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        
        bucket_name = os.environ.get(
            "AWS_BUCKET_NAME", "your-production-bucket-name")
        base_s3_folder = f"renders/{user_id}/{job_id}"

        # Agnostic view file names matching the generated crops
        files_to_upload = [
            ("input_grid.mp4", grid_video_path),
            ("view_0.mp4", os.path.join(INPUT_DIR, "view_0.mp4")),
            ("view_1.mp4", os.path.join(INPUT_DIR, "view_1.mp4")),
            ("view_2.mp4", os.path.join(INPUT_DIR, "view_2.mp4")),
            ("view_3.mp4", os.path.join(INPUT_DIR, "view_3.mp4")),
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