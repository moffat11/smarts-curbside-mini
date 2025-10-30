import cv2, os, imageio, imageio_ffmpeg as ioff
import argparse
import numpy as np

def open_writer(path, fps, W, H):
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264",
        quality=None,  # use bitrate
        bitrate="2000k", pixelformat="yuv420p"
    )
    return writer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="det_track/sample_video.mp4")
    ap.add_argument("--outdir", default="robustness")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cap = cv2.VideoCapture(args.source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or args.fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    dark_w = open_writer(os.path.join(args.outdir, "dark.mp4"), args.fps, W, H)
    blur_w = open_writer(os.path.join(args.outdir, "blur.mp4"), args.fps, 1280, int(1280*H/W))
    lr_w   = open_writer(os.path.join(args.outdir, "lowres.mp4"), args.fps, 854, int(854*H/W))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # ---- dark/contrast ----
        dark = cv2.convertScaleAbs(frame, alpha=1.1, beta=-12)  # contrast↑, brightness↓
        dark_w.append_data(cv2.cvtColor(dark, cv2.COLOR_BGR2RGB))

        # ---- blur ----
        blur = cv2.GaussianBlur(frame, (0,0), 2.0)
        blur = cv2.resize(blur, (1280, int(1280*H/W)))
        blur_w.append_data(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))

        # ---- low resolution ----
        lr = cv2.resize(frame, (854, int(854*H/W)))
        lr_w.append_data(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))

    cap.release()
    dark_w.close(); blur_w.close(); lr_w.close()
    print("Wrote:", os.path.join(args.outdir, "dark.mp4"))
    print("Wrote:", os.path.join(args.outdir, "blur.mp4"))
    print("Wrote:", os.path.join(args.outdir, "lowres.mp4"))

if __name__ == "__main__":
    # force imageio-ffmpeg to use its own bundled ffmpeg binary (avoids DLL issues)
    _ = ioff.get_ffmpeg_exe()
    main()