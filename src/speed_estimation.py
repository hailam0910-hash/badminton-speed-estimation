from ultralytics import YOLO
import cv2
import math
import os

model = YOLO("runs/detect/train/weights/best.pt")

video_path = "videos/video1.mp4"
output_path = "output/speed_output.avi"

os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không mở được video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

ret, frame = cap.read()
if not ret:
    print("Không đọc được frame đầu tiên")
    exit()

h, w = frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# ==== chỉnh scale sau ====
PIXEL_TO_METER = 0.005

prev_center = None
max_speed_kmh = 0

while True:
    results = model(frame, conf=0.2, imgsz=1280, verbose=False)

    current_speed_kmh = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if prev_center is not None:
                dist_pixel = math.sqrt(
                    (cx - prev_center[0]) ** 2 +
                    (cy - prev_center[1]) ** 2
                )

                speed_mps = dist_pixel * PIXEL_TO_METER * fps
                current_speed_kmh = speed_mps * 3.6

                if current_speed_kmh > max_speed_kmh:
                    max_speed_kmh = current_speed_kmh

            prev_center = (cx, cy)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.putText(frame, f"Current Speed: {current_speed_kmh:.2f} km/h",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.putText(frame, f"Max Speed: {max_speed_kmh:.2f} km/h",
                (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    out.write(frame)

    cv2.imshow("Speed Estimation", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    ret, frame = cap.read()
    if not ret:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done:", output_path)
print(f"Max speed: {max_speed_kmh:.2f} km/h")