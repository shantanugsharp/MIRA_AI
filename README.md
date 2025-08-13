# MIRA_AI: Audio-Visual Lip Sync + Face Restoration + Upscaling

This README walks you step-by-step:
1. Connect to the SSH server
2. Set up the environment
3. Run inference (direct Python calls) OR use the pipeline script (run_final.sh)
4. (Optional) Launch the web UI (demo_app.py)
5. Understand required checkpoints and key files
6. Learn the processing flow (how lip sync output is produced)
7. Troubleshooting & tips

---

## 1. Connect to the Server (SSH + (Optional) Port Forwarding)

From your local machine:
```bash
ssh USER@SERVER_IP
# If a custom SSH port:
ssh -p 2222 USER@SERVER_IP
```

If you want to access the Gradio UI locally (tunneling the UI port 7860):
```bash
ssh -L 7860:localhost:7860 USER@SERVER_IP
# Then on server: python demo_app.py
# Open locally: http://localhost:7860
```

If you’ll run long jobs, consider tmux:
```bash
tmux new -s mira
# (Run commands, detach with Ctrl+b then d, reattach:)
tmux attach -t mira
```

---

## 2. Environment Setup

Clone / pull code (if not already present):
```bash
cd /srv/MIRA_AI
# git clone <repo_url> (if needed)
```

Create & activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install core dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
# Submodule / nested projects (if needed):
pip install -r GFPGAN/requirements.txt
pip install -r Real-ESRGAN/requirements.txt
```

Install ffmpeg (required):
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

(Optional UI / extras):
```bash
pip install gradio opencv-python
```

Check GPU (if using CUDA):
```bash
python -c "import torch;print('CUDA:', torch.cuda.is_available())"
nvidia-smi
```

---

## 3. Required Checkpoints (Place Before Running)

Put these in the specified paths (adjust names if different):

| Purpose              | Expected Path / Example                               |
|----------------------|--------------------------------------------------------|
| Wav2Lip model        | checkpoints/Wav2Lip-SD-GAN.pth                         |
| SyncNet (if used)    | checkpoints/syncnet.pth (or equivalent)                |
| GFPGAN               | GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth    |
| Real-ESRGAN (opt)    | Real-ESRGAN/experiments/pretrained_models/*.pth        |
| Face detection       | face_detection/weights/*.pth (as required)             |

Verify:
```bash
ls -1 checkpoints
```

---

## 4. Running Inference (Direct Commands)

Typical inputs:
- Source face video (or single image)
- Target audio (.wav or .mp3)

Convert mp3 to wav if needed:
```bash
ffmpeg -i input/audio.mp3 -ar 16000 -ac 1 input/audio.wav
```

Example (adjust to match your inference.py signature):
```bash
python inference.py \
  --face input/face_video.mp4 \
  --audio input/audio.wav \
  --checkpoint_path checkpoints/Wav2Lip-SD-GAN.pth \
  --outfile results/wav2lip_raw.mp4
```

If you support a single image + audio:
```bash
python inference.py \
  --face input/face_image.jpg \
  --audio input/audio.wav \
  --fps 25 \
  --checkpoint_path checkpoints/Wav2Lip-SD-GAN.pth \
  --outfile results/wav2lip_raw_from_image.mp4
```

(Flags like batch size, pads, resize, etc. depend on your existing script.)

GFPGAN post-processing (per-frame restoration workflow if not automated):
```bash
mkdir -p temp/gfpgan_frames
ffmpeg -i results/wav2lip_raw.mp4 temp/gfpgan_frames/frame_%05d.png
python GFPGAN/inference_gfpgan.py \
  -i temp/gfpgan_frames \
  -o temp/gfpgan_out \
  -v 1.3 -s 2 --only_center_face
ffmpeg -framerate 25 -i temp/gfpgan_out/restored_imgs/frame_%05d.png \
  -i input/audio.wav -c:v libx264 -pix_fmt yuv420p -c:a aac \
  results/final_restored.mp4
```

(Optional) Real-ESRGAN upscale final video frames before muxing audio.

---

## 5. Running the End-to-End Script (run_final.sh)

If the project includes a pipeline script (your repo lists run_final.sh):
```bash
bash run_final.sh \
  --face input/face_video.mp4 \
  --audio input/audio.wav \
  --out results/final_pipeline.mp4
```

If your script is named differently (e.g., final_run.sh), correct the filename:
```bash
bash final_run.sh ...
```

Open the script to see adjustable variables:
```bash
sed -n '1,120p' run_final.sh
```

---

## 6. (Optional) Launch the Web Demo UI

Start (inside venv):
```bash
python demo_app.py
# Visit: http://localhost:7860
```

If remote & you need a shareable tunnel (quick):
```bash
python demo_app.py &
./cloudflared tunnel --url http://localhost:7860
# OR
ngrok http 7860
```

To allow LAN access:
```bash
python demo_app.py --host 0.0.0.0
```

If auth flags were added (example):
```bash
python demo_app.py --host 0.0.0.0 --auth admin:Secret123
```

UI Features (as implemented):
- Upload: video OR image + audio (wav/mp3)
- Toggle GFPGAN restoration (scale, version, center-face)
- Live logs (if streaming mode added)
- Saves outputs under results/

---

## 7. Processing Flow (End-to-End Lip Sync)

1. Input Acquisition  
   - User supplies face video (preferred) OR single image (static) + target speech audio.

2. Preprocessing  
   - Extract video frames (ffmpeg or inside inference).
   - Resample audio (e.g., 16 kHz mono).
   - Detect / align faces (face_detection module / cropping logic).
   - (Optional) Face parsing for masks.

3. Audio Feature Extraction  
   - Convert raw waveform → mel spectrogram (normalization + windowing).

4. Model Inference (Wav2Lip)  
   - For each frame (or sequence), model ingests face crop(s) + audio features.
   - Predicts corrected mouth region or full frame with synced lips.

5. Frame Assembly  
   - Predicted frames (or blended patches) reunited into a video stream (ffmpeg).

6. (Optional) GFPGAN Face Restoration  
   - Extract frames → restore faces (quality, sharpness, realism) → reassemble video.

7. (Optional) Super-Resolution (Real-ESRGAN)  
   - Apply to frames pre- or post-restoration for HD upscale.

8. Audio Muxing  
   - Combine generated frames with the original target audio (synchronization preserved).

9. Output  
   - Final MP4 saved to results/. Intermediate caches in temp/ or frames_* directories.

---

## 8. Important Directories & Files

| Path / File                      | Purpose |
|----------------------------------|---------|
| inference.py                     | Core inference entry point for Wav2Lip |
| wav2lip_train.py / hq_wav2lip_*  | Training scripts |
| models/wav2lip.py                | Wav2Lip architecture |
| models/syncnet.py                | SyncNet (auxiliary sync scoring) |
| face_detection/                  | Face detector utilities & weights |
| GFPGAN/                          | Face restoration code + inference_gfpgan.py |
| Real-ESRGAN/                     | Super-resolution model & scripts |
| run_final.sh                     | Orchestrated full pipeline script |
| demo_app.py                      | Gradio UI (upload + pipeline execution) |
| checkpoints/                     | Holds all required .pth model files |
| results/                         | Final outputs |
| temp/                            | Intermediate frames & working files |
| evaluation/                      | Scoring, metrics, video generation helpers |
| requirements.txt                 | Python dependencies |

---

## 9. Quality & Performance Tips

- GPU Memory: Reduce batch size if OOM.
- Audio Quality: Ensure clean, normalized audio for better lip accuracy.
- Frame Rate: For single image mode, set a sensible FPS (e.g., 25) to avoid choppy output.
- GFPGAN Scale: Common values: 1 or 2. Only integers accepted by its CLI.
- Logging: Use tmux or redirect logs:
  ```bash
  python inference.py ... 2>&1 | tee logs/run_$(date +%s).log
  ```

---

## 10. Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Missing checkpoint error | Wrong path/name | ls checkpoints/ and update args |
| GFPGAN: invalid int value '2.0' | Float passed to -s | Use integer (2) |
| Desync video/audio | Wrong fps or re-encoding | Specify consistent -r in ffmpeg |
| CUDA OOM | VRAM limit | Lower resolution / batch size |
| mp3 not accepted | No decoder | Install ffmpeg, convert to wav |
| UI not reachable | Port/firewall | sudo ufw allow 7860/tcp or use tunnel |

Quick ffmpeg reconversion:
```bash
ffmpeg -i input.mp4 -pix_fmt yuv420p -c:a aac fixed.mp4
```

---

## 11. Example Full Manual Flow (Video + Audio → Final Restored)

```bash
# 1. Run base Wav2Lip
python inference.py --face input/actor.mp4 --audio input/line.wav \
  --checkpoint_path checkpoints/Wav2Lip-SD-GAN.pth \
  --outfile temp/wav2lip_raw.mp4

# 2. Restore faces with GFPGAN
mkdir -p temp/gfp_frames
ffmpeg -i temp/wav2lip_raw.mp4 temp/gfp_frames/frame_%05d.png
python GFPGAN/inference_gfpgan.py -i temp/gfp_frames -o temp/gfp_restored \
  -v 1.3 -s 2 --only_center_face

# 3. Reassemble + mux audio
ffmpeg -framerate 25 -i temp/gfp_restored/restored_imgs/frame_%05d.png \
  -i input/line.wav -c:v libx264 -pix_fmt yuv420p -c:a aac results/final_restored.mp4
```

---

## 12. Roadmap Ideas (Optional)

- Add automated model downloader
- Add Real-ESRGAN toggle in UI
- Add size/rate limiting on uploads
- Batch inference CLI
- Docker + compose for reproducible deployment

---

## 13. Citation / Credits

This project integrates:
- Wav2Lip (lip-sync)
- GFPGAN (face restoration)
- Real-ESRGAN (super-resolution)

Respect original licenses when redistributing models.

---

## 14. Quick Start TL;DR

```bash
# SSH & setup
ssh USER@SERVER
cd /srv/MIRA_AI
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt gradio
# Place checkpoints in checkpoints/

# Fast test
python inference.py --face input/face.mp4 --audio input/audio.wav \
  --checkpoint_path checkpoints/Wav2Lip-SD-GAN.pth \
  --outfile results/test.mp4

# UI
python demo_app.py
```

---

Questions or need automation scripts? Open an issue or ask internally.
