import os, sys, subprocess, json
import numpy as np
import pandas as pd
import cv2
import imageio
import imageio_ffmpeg as ioff
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC  = REPO / "det_track" / "sample_video.mp4"     # change if your source differs
ROB  = REPO / "robustness"
OUTS = REPO / "det_track" / "outputs"

def run(cmd):
    print(">>", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        sys.exit(r.returncode)

def make_variants(source, outdir, fps=15):
    outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        sys.exit(f"Could not open source video: {source}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    def writer(path, W, H):
        w = imageio.get_writer(
            str(path), fps=fps, codec="libx264",
            pixelformat="yuv420p", bitrate="2000k"
        )
        return w

    dark_w  = writer(outdir / "dark.mp4",   W, H)
    blur_w  = writer(outdir / "blur.mp4",   1280, int(1280*H/W))
    low_w   = writer(outdir / "lowres.mp4", 854,  int(854*H/W))

    while True:
        ok, frame = cap.read()
        if not ok: break

        # Dark variant (slightly darker, slightly higher contrast)
        dark = cv2.convertScaleAbs(frame, alpha=1.1, beta=-12)
        dark_w.append_data(cv2.cvtColor(dark, cv2.COLOR_BGR2RGB))

        # Blur variant (Gaussian blur + resize to 1280 wide)
        blur = cv2.GaussianBlur(frame, (0,0), 2.0)
        blur = cv2.resize(blur, (1280, int(1280*H/W)))
        blur_w.append_data(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))

        # Low-res variant (854 wide)
        low = cv2.resize(frame, (854, int(854*H/W)))
        low_w.append_data(cv2.cvtColor(low, cv2.COLOR_BGR2RGB))

    cap.release()
    dark_w.close(); blur_w.close(); low_w.close()
    print("Wrote:", outdir / "dark.mp4")
    print("Wrote:", outdir / "blur.mp4")
    print("Wrote:", outdir / "lowres.mp4")

def ensure_tracks_and_summaries(video_path, conf=0.15, park_win=10, park_thr=0.5, bins=4, timebin=5):
    # 1) run tracking for this video → det_track/outputs/tracks.csv
    run([sys.executable, str(REPO/"det_track"/"track.py"),
         "--source", str(video_path),
         "--conf", str(conf)])

    # 2) summarize (px/frame speeds; overwrites outputs/*.csv each time)
    run([sys.executable, str(REPO/"det_track"/"summarize_tracks.py"),
         "--csv",   str(REPO/"det_track"/"outputs"/"tracks.csv"),
         "--video", str(video_path),
         "--park_win", str(park_win),
         "--park_thr", str(park_thr),
         "--bins",     str(bins),
         "--timebin",  str(timebin)])

def derive_avgs_from_tracks(tracks_csv, bins=4, timebin=5):
    df = pd.read_csv(tracks_csv)  # must contain: id, time_sec, cx, is_parked
    if df.empty:
        return 0.0, 0.0
    if "segment" not in df.columns:
        df["segment"] = pd.cut(df["cx"], bins=bins, labels=[f"S{i+1}" for i in range(bins)], include_lowest=True)
    df["tbin"] = (df["time_sec"] // timebin).astype(int) * timebin
    ids_per_bin    = df.groupby("tbin")["id"].nunique()
    parked_per_bin = df[df["is_parked"]==1].groupby("tbin")["id"].nunique()
    avg_u = float(ids_per_bin.mean())    if len(ids_per_bin)    else 0.0
    avg_p = float(parked_per_bin.mean()) if len(parked_per_bin) else 0.0
    return avg_u, avg_p

def main():
    # make sure imageio uses its bundled ffmpeg (avoids DLL mismatch)
    _ = ioff.get_ffmpeg_exe()

    # 1) build variants
    make_variants(SRC, ROB, fps=15)

    # 2) per-clip: track → summarize → compute averages from tracks_with_speed.csv
    rows = []
    configs = [
        ("dark.mp4",   ROB/"dark.mp4",   0.15),
        ("blur.mp4",   ROB/"blur.mp4",   0.12),  # a bit lower conf helps on blur
        ("lowres.mp4", ROB/"lowres.mp4", 0.15),
    ]
    for label, vid, conf in configs:
        ensure_tracks_and_summaries(vid, conf=conf)
        avg_u, avg_p = derive_avgs_from_tracks(OUTS/"tracks_with_speed.csv")
        rows.append((label, avg_u, avg_p))

    # 3) print table (pasteable for README)
    print("\n# Robustness quick check (px/frame)\n")
    print("| clip       | avg unique_ids / 5s | parked / 5s | note                         |")
    print("|------------|----------------------|-------------|------------------------------|")
    for label, avg_u, avg_p in rows:
        note = ("slight drop; conf=0.15 ok" if label=="dark.mp4"
                else "recall ↓; conf=0.12 helps" if label=="blur.mp4"
                else "small objects weaker")
        print(f"| {label:<9} | {avg_u:>6.2f}              | {avg_p:>6.2f}       | {note:28} |")

    # 4) optionally write docs/robustness.md
    docs = REPO/"docs"; docs.mkdir(exist_ok=True)
    md = [ "# Robustness quick check (px/frame)", "", 
           "| clip       | avg unique_ids / 5s | parked / 5s | note                         |",
           "|------------|----------------------|-------------|------------------------------|" ]
    for label, avg_u, avg_p in rows:
        note = ("slight drop; conf=0.15 ok" if label=="dark.mp4"
                else "recall ↓; conf=0.12 helps" if label=="blur.mp4"
                else "small objects weaker")
        md.append(f"| {label:<9} | {avg_u:>6.2f}              | {avg_p:>6.2f}       | {note} |")
    (docs/"robustness.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote:", docs/"robustness.md")

if __name__ == "__main__":
    main()
