#!/usr/bin/env python3
import cv2, numpy as np, argparse, os

def make_mask(w, h):
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    d = np.maximum(np.abs(x), np.abs(y))
    m = 1.0 - np.clip((d - 0.6) / (1.0 - 0.6), 0, 1)
    m = cv2.GaussianBlur(m, (9,9), 2)
    return m.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="Path to input video (your Wav2Lip or restored mp4)")
    ap.add_argument("--output", required=True, help="Path to output demo video")
    ap.add_argument("--duration", type=float, default=6.0, help="Seconds to render (default 6)")
    ap.add_argument("--blink_at_sec", type=float, default=2.0, help="Blink start time (sec)")
    ap.add_argument("--blink_len", type=int, default=7, help="Total frames in blink")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    assert cap.isOpened(), f"Cannot open video: {args.input}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(n, int(fps * args.duration))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    # Approx eye boxes assuming centered face (quick demo — for prod use landmarks)
    eye_w = int(0.16 * w); eye_h = int(0.10 * h)
    left_cx  = int(0.36 * w); right_cx = int(0.64 * w); eye_cy = int(0.38 * h)
    left_rect  = [left_cx - eye_w//2,  eye_cy - eye_h//2, eye_w, eye_h]
    right_rect = [right_cx - eye_w//2, eye_cy - eye_h//2, eye_w, eye_h]
    eye_mask = make_mask(eye_w, eye_h)

    blink_start = int(args.blink_at_sec * fps)
    blink_len   = max(3, args.blink_len)

    def blink_alpha(fidx):
        if fidx < blink_start or fidx >= blink_start + blink_len: return 0.0
        local = fidx - blink_start
        half  = blink_len // 2
        if local <= half:  a = local / max(1, half)
        else:              a = (blink_len - local) / max(1, blink_len - half)
        return float(np.clip(a, 0.0, 1.0))

    for i in range(max_frames):
        ok, frame = cap.read()
        if not ok: break

        alpha = blink_alpha(i) * 0.55  # subtle
        if alpha > 0.0:
            overlay = frame.copy()
            for (x, y, ew, eh) in (left_rect, right_rect):
                x0 = max(0, x); y0 = max(0, y); x1 = min(w, x+ew); y1 = min(h, y+eh)
                roi = frame[y0:y1, x0:x1].copy()
                if roi.size == 0: continue
                # sample near-eye skin tone for a natural lid color
                sx0 = max(0, x0-8); sy0 = max(0, y0-8); sx1 = min(w, x1+8); sy1 = min(h, y1+8)
                sample = frame[sy0:sy1, sx0:sx1]
                if sample.size == 0:
                    med = np.median(roi.reshape(-1,3), axis=0).astype(np.uint8)
                else:
                    med = np.array([np.median(sample[:,:,c]) for c in range(3)], dtype=np.uint8)
                fill = np.zeros_like(roi); fill[:] = med
                m = eye_mask[:roi.shape[0], :roi.shape[1]][..., None]
                blended = (fill.astype(np.float32)*m + roi.astype(np.float32)*(1.0-m)).astype(np.uint8)
                overlay[y0:y1, x0:x1] = blended
            frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)

        out.write(frame)

    cap.release(); out.release()
    print(f"Demo written to: {args.output}")

if __name__ == "__main__":
    main()
