import os
import shutil
import subprocess
import struct
import urllib.request
import runpod
import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError
from plyfile import PlyData

# --- Workspace Configuration ---
WORKSPACE = "/workspace"
INPUT_DIR = os.path.join(WORKSPACE, "inputs")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")


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

    # 1. Strictly extract SpicyGen's defined payload variables
    video_url = job_input.get("video_url")
    user_id = job_input.get("user_id")
    job_id = job_input.get("job_id")  # <-- Now pulled from SpicyGen's input
    
    # Validation: Enforce strict payload requirements
    if not video_url or not user_id or not job_id:
        return {"error": "Missing required payload data. Must include video_url, user_id, and job_id."}

    cleanup_workspace()
    
    grid_video_path = os.path.join(INPUT_DIR, "input_grid.mp4")
    splat_output = os.path.join(OUTPUT_DIR, "scene_model_4d.splat")

    try:
        print(f"[{job_id}] Downloading source video...")
        urllib.request.urlretrieve(video_url, grid_video_path)

        print(f"[{job_id}] Slicing grid via FFmpeg...")
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{grid_video_path}" -filter_complex '
            f'"[0:v]crop=iw/2:ih/2:0:0[tl]; [0:v]crop=iw/2:ih/2:iw/2:0[tr]; '
            f'[0:v]crop=iw/2:ih/2:0:ih/2[bl]; [0:v]crop=iw/2:ih/2:iw/2:ih/2[br]" '
            f'-map "[tl]" "{INPUT_DIR}/view_front.mp4" '
            f'-map "[tr]" "{INPUT_DIR}/view_back.mp4" '
            f'-map "[bl]" "{INPUT_DIR}/view_left.mp4" '
            f'-map "[br]" "{INPUT_DIR}/view_right.mp4"'
        )
        subprocess.run(ffmpeg_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        print(f"[{job_id}] Running MAtCha initialization...")
        matcha_cmd = (
            f"python3 /workspace/MAtCha/scripts/process_video.py "
            f"--video_dir {INPUT_DIR} "
            f"--output_dir {OUTPUT_DIR}/init_points"
        )
        subprocess.run(matcha_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        print(f"[{job_id}] Running 4C4D optimization (1500 iterations)...")
        c4d_cmd = (
            f"python3 /workspace/4C4D/train.py "
            f"--source_path {INPUT_DIR} "
            f"--model_path {OUTPUT_DIR}/4c4d_model "
            f"--init_pt_cloud {OUTPUT_DIR}/init_points/point_cloud.ply "
            f"--iterations 1500"
        )
        subprocess.run(c4d_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        print(f"[{job_id}] Converting PLY to binary SPLAT format...")
        ply_input = os.path.join(OUTPUT_DIR, "4c4d_model", "point_cloud", "iteration_1500", "point_cloud.ply") 
        convert_ply_to_splat(ply_input, splat_output)
        
        # --- Batch S3 Upload Logic using SpicyGen's IDs ---
        print(f"[{job_id}] Uploading all assets to AWS S3...")
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        
        bucket_name = os.environ.get("AWS_BUCKET_NAME", "your-production-bucket-name")

        # Use SpicyGen's defined job_id for the cloud storage structure
        base_s3_folder = f"renders/{user_id}/{job_id}"

        files_to_upload = [
            ("input_grid.mp4", grid_video_path),
            ("view_front.mp4", os.path.join(INPUT_DIR, "view_front.mp4")),
            ("view_back.mp4", os.path.join(INPUT_DIR, "view_back.mp4")),
            ("view_left.mp4", os.path.join(INPUT_DIR, "view_left.mp4")),
            ("view_right.mp4", os.path.join(INPUT_DIR, "view_right.mp4")),
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
        error_msg = e.stderr.decode() if e.stderr else "Subprocess execution failed"
        return {"error": f"Pipeline error: {error_msg}"}
    except NoCredentialsError:
        return {"error": "AWS credentials not available in RunPod environment secrets."}
    except Exception as e:
        return {"error": f"Internal Process Failure: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})