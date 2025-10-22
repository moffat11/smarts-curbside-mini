import cv2, numpy as np, os

os.makedirs("ocr/samples", exist_ok=True)

def make_sign(path, text_lines, blur=0, brightness=0):
    H, W = 320, 540
    img = np.full((H, W, 3), 255, np.uint8)
    # adjust brightness (positive=brighter, negative=darker)
    img = cv2.convertScaleAbs(img, alpha=1.0, beta=brightness)
    y = 90
    for t, scale, thick in text_lines:
        cv2.putText(img, t, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick, cv2.LINE_AA)
        y += int(60*scale)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0,0), blur)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print("Wrote", path)

# Clean, sharp
make_sign(
    "ocr/samples/sign1.png",
    [("PARKING 2H", 1.6, 3), ("MON-FRI", 1.4, 3), ("9AM-6PM  ->", 1.4, 3)],
    blur=0, brightness=0
)

# Slight blur
make_sign(
    "ocr/samples/sign2.png",
    [("PARKING 2 HOUR", 1.4, 3), ("MON–SAT", 1.2, 3), ("08:00-18:00 ->", 1.2, 3)],
    blur=0.8, brightness=0
)

# Darker + a bit more blur
make_sign(
    "ocr/samples/sign3.png",
    [("PARKING 1H", 1.6, 3), ("TUE-THU", 1.4, 3), ("10AM-4PM ->", 1.4, 3)],
    blur=1.2, brightness=-20
)
