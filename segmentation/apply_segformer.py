# segmentation/apply_segformer.py
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import torch, numpy as np, os, cv2, argparse

LABELS = ["background","road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
          "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"]

def colorize(mask):
    np.random.seed(0)
    palette = (np.random.rand(len(LABELS),3)*255).astype(np.uint8)
    return palette[mask % len(LABELS)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="det_track/sample_video.mp4")
    ap.add_argument("--outdir", default="segmentation/outputs")
    ap.add_argument("--frames", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    feat = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model.eval()

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    step = max(1, int(fps))  # ~1 fps
    i, saved = 0, 0
    while saved < args.frames:
        ok, frame = cap.read()
        if not ok: break
        i += 1
        if i % step: continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = feat(images=rgb, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        up = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        pred = up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        overlay = (0.6*rgb + 0.4*colorize(pred)).astype(np.uint8)
        out = np.hstack([rgb, overlay])
        outp = os.path.join(args.outdir, f"seg_{saved+1:02d}.png")
        Image.fromarray(out).save(outp)
        print("Wrote", outp); saved += 1
    cap.release()