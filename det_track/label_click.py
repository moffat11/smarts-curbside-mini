# det_track/label_click.py
import cv2, csv, os, argparse
from collections import defaultdict

def draw_boxes(img, boxes, color=(0,255,0)):
    out = img.copy()
    for (x1,y1,x2,y2,cls) in boxes:
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="det_track/sample_video.mp4")
    ap.add_argument("--frames", type=int, default=50, help="max frames to label from start")
    ap.add_argument("--out", default="det_track/outputs/labels.csv")
    ap.add_argument("--default_cls", type=int, default=2, help="default class id to store (e.g., car=2)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    maxf = min(args.frames, total) if total else args.frames

    # frame index is 1-based to match your detect CSV convention
    frame_idx = 1
    frame_cache = {}  # frame_idx -> image
    labels = defaultdict(list)  # frame_idx -> list[(x1,y1,x2,y2,cls)]
    tl = None  # top-left of current box

    def load_frame(i):
        if i in frame_cache:
            return frame_cache[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, i-1)
        ok, frame = cap.read()
        if not ok:
            return None
        frame_cache[i] = frame
        return frame

    def on_mouse(event, x, y, flags, param):
        nonlocal tl, frame_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            tl = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and tl:
            x1, y1 = tl; x2, y2 = x, y
            x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
            labels[frame_idx].append((x1, y1, x2, y2, args.default_cls))
            tl = None

    win = "label (n=next, p=prev, z=undo, c=clear, s=save, Esc=save&exit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while 1 <= frame_idx <= maxf:
        frame = load_frame(frame_idx)
        if frame is None:
            break
        disp = draw_boxes(frame, labels[frame_idx])
        cv2.putText(disp, f"frame {frame_idx}/{maxf}  boxes:{len(labels[frame_idx])}", (12,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(disp, f"frame {frame_idx}/{maxf}  boxes:{len(labels[frame_idx])}", (12,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF

        if k == ord('n'):
            frame_idx = min(maxf, frame_idx + 1)
        elif k == ord('p'):
            frame_idx = max(1, frame_idx - 1)
        elif k == ord('z'):
            if labels[frame_idx]:
                labels[frame_idx].pop()
        elif k == ord('c'):
            labels[frame_idx].clear()
        elif k == ord('s') or k == 27:  # s or Esc → save & exit
            break

    cap.release()
    cv2.destroyAllWindows()

    # write CSV
    rows = []
    for f in sorted(labels.keys()):
        for (x1,y1,x2,y2,cls) in labels[f]:
            rows.append({"frame": f, "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2, "cls": cls})
    with open(args.out, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=["frame","xmin","ymin","xmax","ymax","cls"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved {len(rows)} boxes to {args.out}")

if __name__ == "__main__":
    main()
