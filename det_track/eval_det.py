# det_track/eval_det.py
import pandas as pd, numpy as np, argparse, json, os

def iou_xyxy(a, b):
    # a,b = [xmin,ymin,xmax,ymax]
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return 0.0 if ua <= 0 else inter/ua

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="det_track/outputs/detect50.csv", help="from detect_to_csv.py")
    ap.add_argument("--gt_csv",   default="det_track/outputs/labels.csv", help="from label_click.py")
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--out_json", default="det_track/outputs/metrics.json")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv)  # frame,xmin,ymin,xmax,ymax,conf,cls
    gt   = pd.read_csv(args.gt_csv)    # frame,xmin,ymin,xmax,ymax,cls

    if pred.empty or gt.empty:
        raise SystemExit("Empty CSV(s). Ensure you ran detect_to_csv.py and label_click.py.")

    tp, fp, fn = 0, 0, 0
    ious = []

    for f in sorted(gt["frame"].unique()):
        gt_f = gt[gt.frame == f].copy().reset_index(drop=True)
        pr_f = pred[pred.frame == f].copy()
        if pr_f.empty:
            fn += len(gt_f); continue

        pr_f = pr_f.sort_values("conf", ascending=False).reset_index(drop=True)
        used = np.zeros(len(gt_f), dtype=bool)

        for _, r in pr_f.iterrows():
            pb = [r.xmin, r.ymin, r.xmax, r.ymax]
            best_j, best_iou = -1, 0.0
            for j, g in gt_f.iterrows():
                if used[j]: continue
                gb = [g.xmin, g.ymin, g.xmax, g.ymax]
                iou = iou_xyxy(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= args.iou_thr:
                tp += 1
                used[best_j] = True
                ious.append(best_iou)
            else:
                fp += 1

        fn += int((~used).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    mean_iou  = float(np.mean(ious)) if ious else 0.0

    out = {
        "iou_thr": args.iou_thr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": precision, "recall": recall,
        "mean_iou_on_tps": mean_iou,
        "pred_path": args.pred_csv, "gt_path": args.gt_csv
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote:", args.out_json, "\n", out)

if __name__ == "__main__":
    main()