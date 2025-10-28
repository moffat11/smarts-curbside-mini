# det_track/detect_to_csv.py
from ultralytics import YOLO
from pathlib import Path
import argparse, csv, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="det_track/sample_video.mp4", help="video or image path")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 model weights")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--max_frames", type=int, default=50, help="number of frames to process")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="optional list of class indices to keep (e.g. 2 3 5 7 for vehicles)")
    ap.add_argument("--out_csv", default="det_track/outputs/detect50.csv", help="output CSV path")
    args = ap.parse_args()

    out_dir = Path(args.out_csv).parent
    os.makedirs(out_dir, exist_ok=True)

    model = YOLO(args.weights)

    rows, seen = [], 0
    # stream=True yields results per frame
    for r in model.predict(source=args.source,
                           stream=True,
                           imgsz=args.imgsz,
                           conf=args.conf,
                           iou=args.iou,
                           classes=args.classes,
                           save=False,
                           vid_stride=1):
        seen += 1
        if seen > args.max_frames:
            break

        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else []
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else []

        for (x1, y1, x2, y2), c, p in zip(xyxy, clss, conf):
            rows.append({
                "frame": seen,
                "xmin": float(x1), "ymin": float(y1),
                "xmax": float(x2), "ymax": float(y2),
                "conf": float(p), "cls": int(c)
            })

    if rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["frame", "xmin", "ymin", "xmax", "ymax", "conf", "cls"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print("Wrote:", args.out_csv, f"({len(rows)} detections over {min(seen, args.max_frames)} frames)")
    else:
        print("No detections saved. Check --conf/--classes or your video path.")

if __name__ == "__main__":
    main()
