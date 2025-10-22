import cv2, argparse, os
import pytesseract

# Windows: point to the exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_image(path):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # light cleanup (you can extend later)
    text = pytesseract.image_to_string(gray, config="--psm 6")
    return text

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to a sign image")
    ap.add_argument("--out", default="ocr/outputs/ocr_raw.txt")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    txt = ocr_image(args.image)
    open(args.out, "w", encoding="utf-8").write(txt)
    print("OCR saved to:", args.out)
