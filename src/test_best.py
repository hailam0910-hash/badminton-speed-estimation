from ultralytics import YOLO
import cv2
import os

model_path = "runs/detect/train-2/weights/best.pt"
video_path = "videos/video1.mp4"
output_path = "output/best_output.avi"

os.makedirs("output", exist_ok=True)

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.15, imgsz=1280, verbose=False)
    annotated = results[0].plot()

    out.write(annotated)
    cv2.imshow("Best model", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done:", output_path)