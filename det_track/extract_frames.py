import cv2, os
os.makedirs("det_track/frames", exist_ok=True)
cap = cv2.VideoCapture("det_track/sample_video.mp4")
i = 0
while i < 50:
    ok, frame = cap.read()
    if not ok: break
    i += 1
    out = f"det_track/frames/{i:06d}.jpg"
    cv2.imwrite(out, frame)
cap.release()
print("Wrote", i, "frames to det_track/frames")
