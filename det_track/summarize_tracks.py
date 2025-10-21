# det_track/summarize_tracks.py
import argparse, os
import numpy as np
import pandas as pd
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="det_track/outputs/tracks.csv")
    ap.add_argument("--video", default="det_track/sample_video.mp4", help="Video file to read FPS from")
    ap.add_argument("--park_win", type=int, default=10, help="rolling window (frames) for parked heuristic")
    ap.add_argument("--park_thr", type=float, default=0.5, help="mean speed threshold (px/frame) to mark parked")
    ap.add_argument("--bins", type=int, default=4, help="horizontal segments of frame for simple occupancy")
    ap.add_argument("--timebin", type=float, default=5.0, help="seconds per time bin for aggregation")
    args = ap.parse_args()

    out_dir = "det_track/outputs"
    os.makedirs(out_dir, exist_ok=True)

    # --- read csv
    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("tracks.csv is empty — run track.py first.")

    # --- get FPS to convert frames -> seconds
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    df["time_sec"] = (df["frame"] - 1) / fps

    # --- center points + per-frame speed in pixels
    df["cx"] = (df["xmin"] + df["xmax"]) / 2.0
    df["cy"] = (df["ymin"] + df["ymax"]) / 2.0
    df = df.sort_values(["id", "frame"])
    df["dx"] = df.groupby("id")["cx"].diff().fillna(0.0)
    df["dy"] = df.groupby("id")["cy"].diff().fillna(0.0)
    df["speed_px"] = np.hypot(df["dx"], df["dy"])

    # --- parked vs moving via rolling mean of speed
    df["speed_ma"] = df.groupby("id")["speed_px"].transform(
        lambda s: s.rolling(args.park_win, min_periods=max(2, args.park_win // 2)).mean()
    )
    df["is_parked"] = (df["speed_ma"] < args.park_thr).astype(int)

    out_tracks = os.path.join(out_dir, "tracks_with_speed.csv")
    df.to_csv(out_tracks, index=False)
    print("Wrote:", out_tracks)

    # --- first seen frame/time per ID
    first_seen = (df.groupby("id")
                    .agg(first_seen_frame=("frame", "min"),
                         first_seen_sec=("time_sec", "min"))
                    .reset_index())
    out_first = os.path.join(out_dir, "first_seen_by_id.csv")
    first_seen.to_csv(out_first, index=False)
    print("Wrote:", out_first)

    # --- simple horizontal “segments” across the image
    # derive width from max xmax (fallback to 1920)
    frame_w = float(df["xmax"].max()) if len(df) else 1920.0
    df["segment"] = pd.cut(df["cx"],
                           bins=args.bins,
                           labels=[f"S{i+1}" for i in range(args.bins)],
                           include_lowest=True)

    # --- time bins in seconds (e.g., 5s bins)
    df["tbin"] = (df["time_sec"] // args.timebin).astype(int) * args.timebin

    # unique vehicle count per segment per time bin
    counts = (df.groupby(["segment", "tbin"])["id"]
                .nunique()
                .reset_index(name="unique_ids"))
    out_counts = os.path.join(out_dir, "counts_by_segment.csv")
    counts.to_csv(out_counts, index=False)
    print("Wrote:", out_counts)

if __name__ == "__main__":
    main()
