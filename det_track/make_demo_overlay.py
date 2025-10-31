# Create a short annotated demo clip without ffmpeg CLI.
import cv2, argparse, math
from ultralytics import YOLO

def draw_one(frame, boxes, classes, names):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls = int(b.cls)
        conf = float(b.conf)
        label = f"{names.get(cls, str(cls))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, max(15, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="det_track/sample_video.mp4")
    ap.add_argument("--out", default="det_track/demo_15s.mp4")
    ap.add_argument("--start_sec", type=float, default=0.0)
    ap.add_argument("--duration_sec", type=float, default=15.0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--classes", type=int, nargs="+", default=[2,3,5,7])  # car, motorcycle, bus, truck
    ap.add_argument("--width", type=int, default=960)  # output width
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_f = int(args.start_sec * src_fps)
    end_f   = int((args.start_sec + args.duration_sec) * src_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read first frame at start position")

    h, w = frame.shape[:2]
    out_w = args.width
    out_h = int(round(h * (out_w / float(w))))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit("Failed to open VideoWriter. Try a different FOURCC (e.g., 'XVID') or width/fps.")

    model = YOLO("yolov8n.pt")
    names = model.model.names if hasattr(model, "model") else {}

    frame_idx = start_f
    wrote = 0
    stride = max(1, int(round(src_fps / args.fps)))  # simple frame skip to hit target fps

    while frame_idx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # decimate to target FPS
        if (frame_idx - start_f) % stride:
            continue

        # run YOLO on this frame
        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                            iou=args.iou, classes=args.classes, verbose=False)
        if res and len(res) > 0 and res[0].boxes is not None:
            draw_one(frame, res[0].boxes, args.classes, names)

        # resize and write
        out_frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(out_frame)
        wrote += 1

    cap.release()
    writer.release()
    print(f"Wrote {args.out} ({wrote} frames @ {args.fps} fps)")

if __name__ == "__main__":
    main()