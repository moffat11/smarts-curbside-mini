# Build GeoJSON LineStrings from det_track/outputs/tracks_with_speed.csv
# Coordinates are in image pixels (x=cx, y=cy). QGIS will open this fine;
# treat it as a local/pseudo CRS for visualization.

import json, argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_csv", default="det_track/outputs/tracks_with_speed.csv")
    ap.add_argument("--out_geojson", default="gis/outputs/tracks_lines.geojson")
    ap.add_argument("--min_points", type=int, default=2, help="min points per track to keep")
    args = ap.parse_args()

    tracks_csv = Path(args.tracks_csv)
    out_path = Path(args.out_geojson)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tracks_csv)
    if df.empty:
        raise SystemExit(f"No rows in {tracks_csv}")

    # Ensure we have what we need
    missing = [c for c in ["id", "time_sec", "cx", "cy"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {tracks_csv}: {missing}")

    # Sort consistently
    df = df.sort_values(["id", "time_sec"], kind="mergesort")

    features = []
    for tid, g in df.groupby("id"):
        if len(g) < args.min_points:
            continue
        coords = [[float(x), float(y)] for x, y in zip(g["cx"], g["cy"])]
        props = {
            "id": int(tid),
            "n_pts": int(len(g)),
            "start_time": float(g["time_sec"].iloc[0]),
            "end_time": float(g["time_sec"].iloc[-1]),
            "avg_speed_px": float(g["speed_px"].mean()) if "speed_px" in g else None,
            "avg_speed_ma": float(g["speed_ma"].mean()) if "speed_ma" in g else None,
            "parked_frames": int((g.get("is_parked", 0) == 1).sum()),
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        })

    fc = {
        "type": "FeatureCollection",
        "name": "tracks_lines (image pixels)",
        "features": features,
        # Optional hint: this is a pixel-based local CRS, not lon/lat
        "crs": {"type": "name", "properties": {"name": "EPSG:0 (image-pixels)"}}
    }

    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    print("Wrote:", out_path, f"({len(features)} features)")

if __name__ == "__main__":
    main()