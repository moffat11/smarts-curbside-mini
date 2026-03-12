# Smart Curbside Analytics 🚗🛣️

An end-to-end computer vision pipeline designed to analyze curbside environments, track vehicle movement, estimate speeds, and interpret parking signage. This project utilizes object detection, multi-object tracking, optical character recognition (OCR), and semantic segmentation to extract actionable geospatial data from raw street-level video.

## 🚀 Key Features
* **Vehicle Detection & Tracking:** Utilizes YOLOv8 for robust object detection and tracks multi-object movement across video frames.
* **Speed Estimation & Parking Logic:** Maps pixel-based frame movement to real-world meters using homography, classifying vehicles as parked if their moving average speed drops below 0.2 m/s.
* **Signage OCR:** Extracts and parses text from parking signs using Tesseract OCR to interpret curbside rules.
* **Semantic Segmentation:** Applies a pretrained SegFormer-B0 (Transformer) to extract scene context (roads, vehicles, traffic signs).
* **Geospatial Mapping:** Generates GeoJSON outputs for mapping and spatial analysis within QGIS.

## 🛠️ Tech Stack
* **Computer Vision & ML:** YOLOv8 (Ultralytics), SegFormer (Transformers), OpenCV (`cv2`)
* **OCR:** PyTesseract
* **Data Processing:** Python, Pandas, JSON
* **Mapping:** GeoJSON, QGIS
* **Video Sourcing:** `yt-dlp`

## 🏗️ Core Pipelines

### 1. Object Detection & Tracking (`det_track/`)
* `detect.py`: Runs YOLOv8 over video inputs to draw bounding boxes and save annotated results.
* `track.py`: Executes multi-object tracking and exports frame-by-frame tracks to CSV.
* `summarize_tracks.py`: Reads the tracking CSV to compute per-ID speeds, flag parked vehicles (`is_Parked`), and generate per-segment time-binned counts.

### 2. OCR & Sign Interpretation (`ocr/`)
* `ocr_signs.py`: Processes sign images to extract raw text.
* `parse_rules.py`: Cleans and structures the raw text into logical rules.
* `make_signs.py`: Utility script to generate sample signs for testing.

### 3. Transformer Segmentation
Applied a pretrained SegFormer-B0 (ADE-20K dataset) to sample curbside frames to establish scene context. 
* *Note:* While the ADE-20K model is not curb-specific (lacking strict geometry/curb semantics), it successfully demonstrates Transformer-based segmentation competence for downstream analytics.
* Outputs side-by-side qualitative overlays (e.g., `segmentation/outputs/seg_01.png`).

## 📊 Analytics & Robustness

### Detection Sanity Metrics
Evaluated a lightweight sanity check on the first 50 frames against manual labels:
* Computes Precision and Recall @ IoU 0.5.
* Calculates mean IoU on matched boxes.
* Metrics govern the confidence thresholds, NMS settings, and downstream parked-vehicle thresholds.

### Robustness Testing
Evaluated pipeline resilience against altered video variants (5-second intervals). Adjusted confidence thresholds effectively mitigate accuracy drops in degraded conditions.

| Video Condition | Avg Unique IDs / 5s | Parked / 5s | Notes |
| :--- | :--- | :--- | :--- |
| **Dark (`dark.mp4`)** | 30.83 | 1.00 | Slight drop; lowering `conf=0.15` stabilizes tracking. |
| **Blur (`blur.mp4`)** | 31.58 | 1.00 | Recall decreases; lowering `conf=0.12` improves detection. |
| **Low Res (`lowres.mp4`)** | 31.04 | 1.00 | Small objects become difficult to detect. |

## 💻 How to Reproduce

**1. Run Object Tracking**
```bash
python det_track/track.py --source <video_path>
