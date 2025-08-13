
import gradio as gr
import os
import shutil
import subprocess
import uuid
from datetime import datetime
import cv2
import numpy as np
import threading
import time

INPUT_DIR = "input"
RESULTS_DIR = "results"
TEMP_DIR = "temp"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

WAV2LIP_CKPT = "checkpoints/Wav2Lip-SD-GAN.pth"  # adjust if different
# Path to the GFPGAN repo directory (NOT the inference_gfpgan.py file itself)
GFPGAN_REPO = "/srv/MIRA_AI/GFPGAN"  # contains inference_gfpgan.py

def _save_upload(upload, dst_path, kind="video_or_image"):
    """Save gradio upload (video path, image array, or audio path) to destination.

    kind: one of ["video_or_image", "audio", "image"] used for error messages.
    """
    if upload is None:
        return None
    # If it's a numpy array (image from gr.Image)
    if isinstance(upload, np.ndarray):
        # Assume RGB, convert to BGR for cv2
        bgr = cv2.cvtColor(upload, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst_path, bgr)
        return dst_path
    # If it's a simple string path
    if isinstance(upload, str) and os.path.isfile(upload):
        shutil.copy(upload, dst_path)
        return dst_path
    # File-like object
    if hasattr(upload, 'name') and os.path.isfile(upload.name):
        shutil.copy(upload.name, dst_path)
        return dst_path
    # Dict format from some components
    if isinstance(upload, dict):
        # Audio can be tuple-like; for safety handle 'name'
        name = upload.get('name') if isinstance(upload.get('name'), str) else None
        if name and os.path.isfile(name):
            shutil.copy(name, dst_path)
            return dst_path
    raise ValueError(f"Unsupported {kind} upload format; expected file path or ndarray for image.")

def _save_audio(upload, uid):
    """Save audio upload (wav/mp3 or array) preserving extension.

    Returns the destination absolute path.
    """
    if upload is None:
        raise ValueError("Audio file required.")
    # If gradio supplies a filepath string
    if isinstance(upload, str) and os.path.isfile(upload):
        ext = os.path.splitext(upload)[1].lower() or ".wav"
        dest = os.path.join(INPUT_DIR, f"audio_{uid}{ext}")
        shutil.copy(upload, dest)
        return dest
    # Dict with name
    if isinstance(upload, dict) and 'name' in upload and os.path.isfile(upload['name']):
        src = upload['name']
        ext = os.path.splitext(src)[1].lower() or ".wav"
        dest = os.path.join(INPUT_DIR, f"audio_{uid}{ext}")
        shutil.copy(src, dest)
        return dest
    # Tuple (sr, data) or other raw forms
    if isinstance(upload, (list, tuple)) and len(upload) == 2:
        sr, data = upload
        ext = ".wav"
        dest = os.path.join(INPUT_DIR, f"audio_{uid}{ext}")
        try:
            import soundfile as sf  # type: ignore
            sf.write(dest, data, sr)
        except Exception:
            from scipy.io import wavfile
            data16 = (np.asarray(data) * 32767).astype(np.int16)
            wavfile.write(dest, sr, data16)
        return dest
    raise ValueError("Unsupported audio upload format.")

def run_wav2lip(face_path, audio_path, out_path, fps_for_image=None):
    cmd = [
        "python", "inference.py",
        "--checkpoint_path", WAV2LIP_CKPT,
        "--face", face_path,
        "--audio", audio_path,
        "--outfile", out_path
    ]
    # If input is an image and user supplied fps, append
    if fps_for_image is not None:
        cmd += ["--fps", str(fps_for_image)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Wav2Lip failed:\n{res.stdout}")
    return out_path, res.stdout

def run_gfpgan(input_video, output_dir, scale=2, version="1.3", only_center_face=False):
    """Run face restoration pipeline script on a video."""
    script_path = os.path.join("scripts", "face_restore_pipeline.py")
    if not os.path.isfile(script_path):
        raise FileNotFoundError("face_restore_pipeline.py not found; cannot run GFPGAN pipeline.")
    # Always pass integer scale; GFPGAN upstream expects int for -s/--upscale
    scale_int = int(round(float(scale)))
    cmd = [
        "python", script_path,
        "--input_video", input_video,
        "--output_dir", output_dir,
        "--gfpgan_repo", GFPGAN_REPO,
        "--gfpgan_version", version,
        "--scale", str(scale_int),
    ]
    if only_center_face:
        cmd.append("--only_center_face")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0 and "invalid int value" in res.stdout:
        # Retry once forcing int formatting (defensive) even though we already did
        cmd_retry = cmd
        res = subprocess.run(cmd_retry, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"GFPGAN pipeline failed:\n{res.stdout}")
    final_video = os.path.join(output_dir, "output_restored.mp4")
    if not os.path.isfile(final_video):
        raise RuntimeError("GFPGAN did not produce output_restored.mp4")
    return final_video, res.stdout

def _stream_process(cmd, log_prefix):
    """Spawn a subprocess and yield lines of its combined stdout/stderr in real-time."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            yield f"[{log_prefix}] {line.rstrip()}"
    finally:
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({log_prefix}) with code {proc.returncode}")

def lip_sync_and_restore_stream(video, image, audio, restore_faces, gfpgan_scale, gfpgan_version, only_center_face, image_fps):
    """Streaming variant of the pipeline yielding incremental logs."""
    uid = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    face_is_image = False
    raw_face_video = os.path.join(INPUT_DIR, f"video_{uid}.mp4")
    raw_face_image = os.path.join(INPUT_DIR, f"image_{uid}.png")
    wav2lip_out = os.path.join(TEMP_DIR, f"wav2lip_{uid}.mp4")
    final_out = os.path.join(RESULTS_DIR, f"final_{uid}.mp4")
    logs = []

    def push(msg):
        logs.append(msg)
        # yield placeholder None for video until final; logs joined
        return None, "\n".join(logs)

    try:
        if video is not None:
            _save_upload(video, raw_face_video, kind="video")
            face_input_path = raw_face_video
            push("Video input saved.")
        elif image is not None:
            _save_upload(image, raw_face_image, kind="image")
            face_input_path = raw_face_image
            face_is_image = True
            push("Image input saved.")
        else:
            yield push("ERROR: You must provide either a video or an image.")
            return

        raw_audio = _save_audio(audio, uid)
        push(f"Audio saved: {os.path.basename(raw_audio)}")

        fps_for_image = None
        if face_is_image:
            fps_for_image = int(image_fps) if image_fps and image_fps > 0 else 25
            push(f"Static image mode with fps={fps_for_image}.")

        # Wav2Lip streaming
        cmd = [
            "python", "inference.py",
            "--checkpoint_path", WAV2LIP_CKPT,
            "--face", face_input_path,
            "--audio", raw_audio,
            "--outfile", wav2lip_out
        ]
        if fps_for_image is not None:
            cmd += ["--fps", str(fps_for_image)]
        push("Starting Wav2Lip inference...")
        try:
            for line in _stream_process(cmd, "Wav2Lip"):
                yield None, "\n".join(logs + [line])
        except Exception as e:
            yield push(f"ERROR during Wav2Lip: {e}")
            return
        push("Wav2Lip inference complete.")

        restored_video = wav2lip_out
        # GFPGAN stage (optional)
        if restore_faces:
            gfpgan_dir = os.path.join(TEMP_DIR, f"gfpgan_{uid}")
            os.makedirs(gfpgan_dir, exist_ok=True)
            scale_int = int(round(float(gfpgan_scale)))
            gfp_cmd = [
                "python", os.path.join("scripts", "face_restore_pipeline.py"),
                "--input_video", restored_video,
                "--output_dir", gfpgan_dir,
                "--gfpgan_repo", GFPGAN_REPO,
                "--gfpgan_version", gfpgan_version,
                "--scale", str(scale_int)
            ]
            if only_center_face:
                gfp_cmd.append("--only_center_face")
            push("Starting GFPGAN restoration...")
            try:
                for line in _stream_process(gfp_cmd, "GFPGAN"):
                    yield None, "\n".join(logs + [line])
            except Exception as e:
                yield push(f"ERROR during GFPGAN: {e}")
                return
            push("GFPGAN restoration complete.")
            restored_video = os.path.join(gfpgan_dir, "output_restored.mp4")

        shutil.copy(restored_video, final_out)
        push(f"Final video saved: {final_out}")
        # Final yield with video
        yield final_out, "\n".join(logs)
    except Exception as e:
        yield None, "\n".join(logs + [f"ERROR: {e}"])

def lip_sync_and_restore(video, image, audio, restore_faces, gfpgan_scale, gfpgan_version, only_center_face, image_fps):
    """Full pipeline: Wav2Lip -> (optional) GFPGAN restoration.
    Accepts either a video or an image as face input. If both provided, video takes precedence.
    """
    uid = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    face_is_image = False
    raw_face_video = os.path.join(INPUT_DIR, f"video_{uid}.mp4")
    raw_face_image = os.path.join(INPUT_DIR, f"image_{uid}.png")
    # raw_audio path decided dynamically by _save_audio (keeps extension)
    wav2lip_out = os.path.join(TEMP_DIR, f"wav2lip_{uid}.mp4")
    final_out = os.path.join(RESULTS_DIR, f"final_{uid}.mp4")
    logs = []

    try:
        if video is not None:
            _save_upload(video, raw_face_video, kind="video")
            face_input_path = raw_face_video
            logs.append("Video input saved.")
        elif image is not None:
            _save_upload(image, raw_face_image, kind="image")
            face_input_path = raw_face_image
            face_is_image = True
            logs.append("Image input saved.")
        else:
            raise ValueError("You must provide either a video or an image.")

        raw_audio = _save_audio(audio, uid)
        logs.append(f"Audio saved: {os.path.basename(raw_audio)}")

        fps_for_image = None
        if face_is_image:
            fps_for_image = int(image_fps) if image_fps and image_fps > 0 else 25
            logs.append(f"Static image mode with fps={fps_for_image}.")

        out_vid, wav2lip_log = run_wav2lip(face_input_path, raw_audio, wav2lip_out, fps_for_image=fps_for_image)
        logs.append("Wav2Lip inference complete.")
        logs.append(wav2lip_log)

        restored_video = out_vid
        if restore_faces:
            gfpgan_dir = os.path.join(TEMP_DIR, f"gfpgan_{uid}")
            os.makedirs(gfpgan_dir, exist_ok=True)
            restored_video, gfpgan_log = run_gfpgan(out_vid, gfpgan_dir, scale=gfpgan_scale, version=gfpgan_version, only_center_face=only_center_face)
            logs.append("GFPGAN restoration complete.")
            logs.append(gfpgan_log)

        shutil.copy(restored_video, final_out)
        logs.append(f"Final video saved: {final_out}")
        return final_out, "\n".join(logs)
    except Exception as e:
        logs.append(f"ERROR: {e}")
        return None, "\n".join(logs)

with gr.Blocks(title="MIRA_AI Demo") as demo:
    gr.Markdown("# MIRA_AI Lip Sync + Face Restoration Demo\nUpload either a face video OR a single face image along with an audio track. Optionally apply GFPGAN face restoration.")
    with gr.Row():
        video_in = gr.Video(label="Input Video (.mp4)", interactive=True)
        image_in = gr.Image(label="OR Face Image (png/jpg)", type="numpy")
        audio_in = gr.Audio(label="Input Audio (.wav)")
    with gr.Row():
        image_fps = gr.Slider(minimum=1, maximum=60, value=25, step=1, label="FPS for Static Image (used only if image uploaded)")
    with gr.Accordion("GFPGAN Options", open=False):
        restore_faces = gr.Checkbox(label="Enable GFPGAN Face Restoration", value=True)
        gfpgan_scale = gr.Slider(1, 4, value=2, step=1, label="GFPGAN Scale")
        gfpgan_version = gr.Radio(["1.2", "1.3", "1.4"], value="1.3", label="GFPGAN Version")
        only_center_face = gr.Checkbox(label="Only Center Face", value=False)
    run_btn = gr.Button("Run Pipeline (Buffered Logs)")
    stream_btn = gr.Button("Run Pipeline (Live Logs)")
    with gr.Row():
        output_video = gr.Video(label="Output Video")
        logs_out = gr.Textbox(label="Logs", lines=20)

    run_btn.click(lip_sync_and_restore, inputs=[video_in, image_in, audio_in, restore_faces, gfpgan_scale, gfpgan_version, only_center_face, image_fps], outputs=[output_video, logs_out])
    stream_btn.click(lip_sync_and_restore_stream, inputs=[video_in, image_in, audio_in, restore_faces, gfpgan_scale, gfpgan_version, only_center_face, image_fps], outputs=[output_video, logs_out])

if __name__ == "__main__":
    print("Starting Gradio demo. Ensure checkpoints exist: " + WAV2LIP_CKPT)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
