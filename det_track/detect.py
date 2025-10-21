from ultralytics import YOLO
def main():
    model = YOLO("yolov8n.pt")
    source = "sample_video.mp4"
    results = model(source, save=True, conf=0.25)
    print("Saved to:", results[0].save_dir)
if __name__ == "__main__":
    main()