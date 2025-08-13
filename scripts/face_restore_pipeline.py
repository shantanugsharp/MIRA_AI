#!/usr/bin/env python3
import argparse, os, sys, subprocess, glob, shutil

def run(cmd):
    print(">>", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise SystemExit(f"Command failed: {' '.join(cmd)}")
    return p.stdout

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def detect_fps(video_path):
    # Use ffprobe to get exact FPS
    try:
        out = run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "default=nokey=1:noprint_wrappers=1",
            video_path
        ]).strip()
        if "/" in out:
            num, den = out.split("/")
            return float(num) / float(den) if den != "0" else float(num)
        return float(out)
    except Exception:
        return 25.0

def has_nvenc():
    try:
        out = run(["ffmpeg", "-hide_banner", "-encoders"])
        return "h264_nvenc" in out
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser(description="Face restoration (GFPGAN-only) video pipeline")
    ap.add_argument("--input_video", required=True, help="Path to input video")
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument("--gfpgan_repo", required=True, help="Path to GFPGAN repo (where inference_gfpgan.py lives)")
    ap.add_argument("--gfpgan_version", default="1.3", help="GFPGAN model version (default 1.3)")
    ap.add_argument("--scale", type=float, default=2, help="GFPGAN output scale (default 2)")
    ap.add_argument("--only_center_face", action="store_true", help="Restore only center face (faster)")
    ap.add_argument("--keep_background_upscaler", default="none",
                    choices=["none", "esrgan"], help="Use background upsampler inside GFPGAN (default none)")
    ap.add_argument("--fps", type=float, default=0.0, help="Force FPS; 0=auto from input")
    ap.add_argument("--start_number", type=int, default=1, help="Frame start number for ffmpeg pattern")
    args = ap.parse_args()

    in_video = os.path.abspath(args.input_video)
    out_dir = os.path.abspath(args.output_dir)
    ensure_dir(out_dir)

    frames_dir = ensure_dir(os.path.join(out_dir, "frames_src"))
    restored_dir = os.path.join(out_dir, "restored_imgs")
    # GFPGAN creates restored_imgs under output_dir by default; clean if exists
    if os.path.isdir(restored_dir):
        shutil.rmtree(restored_dir)

    # 1) Extract frames (zero-padded)
    # We keep extraction simple and consistent; fps auto-detected is used later for mux.
    run([
        "ffmpeg", "-y", "-i", in_video,
        os.path.join(frames_dir, "frame_%05d.jpg")
    ])

    # 2) Run GFPGAN over frames
    gfpgan_py = os.path.join(os.path.abspath(args.gfpgan_repo), "inference_gfpgan.py")
    if not os.path.isfile(gfpgan_py):
        raise SystemExit(f"inference_gfpgan.py not found at: {gfpgan_py}")

    # GFPGAN's inference_gfpgan.py expects an integer for -s/--upscale; ensure we cast
    upscale_int = int(round(float(args.scale)))
    gfpgan_cmd = [
        sys.executable, gfpgan_py,
        "-i", frames_dir,
        "-o", out_dir,
        "-v", str(args.gfpgan_version),
        "-s", str(upscale_int),
    ]
    if args.only_center_face:
        gfpgan_cmd.append("--only_center_face")
    # background upsampler
    if args.keep_background_upscaler == "none":
        gfpgan_cmd += ["--bg_upsampler", "None"]
    else:
        # use realesrgan as bg upsampler *inside GFPGAN* (lighter than full ESRGAN pass)
        gfpgan_cmd += ["--bg_upsampler", "realesrgan"]

    run(gfpgan_cmd)

    # 3) Sanity: ensure restored frames exist
    if not os.path.isdir(restored_dir):
        raise SystemExit(f"GFPGAN did not produce {restored_dir}")

    # 4) Mux restored frames back to video with original audio
    fps = args.fps if args.fps > 0 else detect_fps(in_video)
    encoder = "h264_nvenc" if has_nvenc() else "libx264"

    # Detect available images and extension
    pngs = sorted(glob.glob(os.path.join(restored_dir, "*.png")))
    jpgs = sorted(glob.glob(os.path.join(restored_dir, "*.jpg")))
    imgs = pngs if pngs else jpgs
    if not imgs:
        raise SystemExit("No restored images found to mux (looked for *.png and *.jpg).")

    ext = os.path.splitext(imgs[0])[1].lower()  # ".png" or ".jpg"
    # Decide whether we can use a numbered pattern or need glob
    start_num = args.start_number
    # Check if the first expected file exists
    expected_first = os.path.join(restored_dir, f"frame_{start_num:05d}{ext}")
    use_numbered_pattern = os.path.exists(expected_first)

    out_video = os.path.join(out_dir, "output_restored.mp4")

    if use_numbered_pattern:
        # Try numbered pattern with provided start number
        run([
            "ffmpeg", "-y",
            "-framerate", f"{fps}",
            "-start_number", f"{start_num}",
            "-i", os.path.join(restored_dir, f"frame_%05d{ext}"),
            "-i", in_video,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", encoder, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            out_video
        ])
    else:
        # Fall back to glob pattern (robust to naming and start number)
        run([
            "ffmpeg", "-y",
            "-framerate", f"{fps}",
            "-pattern_type", "glob",
            "-i", os.path.join(restored_dir, f"*{ext}"),
            "-i", in_video,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", encoder, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            out_video
        ])

    print(f"Done. Restored video: {out_video}")

if __name__ == "__main__":
    main()
