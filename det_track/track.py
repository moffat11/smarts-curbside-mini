# det_track/track.py
# Run multi-object tracking on a video and export per-frame tracks to CSV.
# Usage (from project root):
#   python det_track/track.py --source "det_track/sample_video.mp4"
# Or (from inside det_track/):
#   python track.py --source "sample_video.mp4"

from ultralytics import YOLO
from pathlib import Path
import argparse, csv, os

# COCO class IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [2, 3, 5, 7]

def main():
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="YOLOv8 + ByteTrack to CSV")
    ap.add_argument("--source", default=str(here / "sample_video.mp4"), help="Path to video file or folder")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--classes", type=int, nargs="*", default=VEHICLE_CLASSES, help="Class IDs to keep")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config (bytetrack.yaml or botsort.yaml)")
    ap.add_argument("--vid_stride", type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--out_csv", default=str(here / "outputs" / "tracks.csv"), help="Where to write CSV")
    args = ap.parse_args()

    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    # Load model
    model = YOLO(args.weights)

    # Run tracking (Ultralytics writes annotated video to runs/track/exp*)
    results = model.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes if args.classes else None,
        tracker=args.tracker,
        persist=True,     # keep IDs across frames
        save=True,        # save annotated video
        vid_stride=args.vid_stride
    )

    # Collect per-frame data
    rows = []
    frame_i = 0

    for r in results:
        frame_i += 1
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        # Safely extract tensors
        xyxy = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
        ids  = boxes.id.cpu().numpy()   if getattr(boxes, "id",   None) is not None else [-1]*len(xyxy)
        clss = boxes.cls.cpu().numpy()  if getattr(boxes, "cls",  None) is not None else [-1]*len(xyxy)
        conf = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else [0]*len(xyxy)

        for (x1, y1, x2, y2), tid, c, p in zip(xyxy, ids, clss, conf):
            rows.append({
                "frame": frame_i,
                "time": frame_i,                # use frame index as time proxy (you can map to seconds later)
                "id": int(tid) if tid is not None else -1,
                "xmin": float(x1), "ymin": float(y1),
                "xmax": float(x2), "ymax": float(y2),
                "conf": float(p),
                "cls": int(c)
            })

    if not rows:
        raise SystemExit("No tracks were produced. Try lowering --conf, removing --classes, or converting your video to .mp4")

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {args.out_csv}")
    print("Annotated video is in runs/track/exp*/ (Ultralytics default)")

if __name__ == "__main__":
    main()
