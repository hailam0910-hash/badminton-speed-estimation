from ultralytics import YOLO
import cv2
import os

video_path = "videos/video1.mp4"
output_path = "output/yolo_output.mp4"

os.makedirs("output", exist_ok=True)

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không mở được video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.15, imgsz=1280, verbose=False)
    annotated = results[0].plot()

    out.write(annotated)

    cv2.imshow("YOLO detect", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done:", output_path)