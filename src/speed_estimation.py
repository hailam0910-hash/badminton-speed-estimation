from ultralytics import YOLO
import cv2
import math
import os

# ===== MODEL =====
model = YOLO("runs/detect/train-2/weights/best.pt")

# ===== VIDEO =====
video_path = "videos/IMG_9806.MOV"

# lấy tên video
video_name = os.path.splitext(
    os.path.basename(video_path))[0]


# output tự động
output_path = f"output/{video_name}_output.avi"

# ===== OUTPUT =====
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

# ===== VIDEO WRITER =====
fourcc = cv2.VideoWriter_fourcc(*"MJPG")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (w, h)
)

# ===== SCALE (TEMP) =====
PIXEL_TO_METER = 0.005

# ===== TRACKING =====
prev_center = None

# ===== SPEED =====
max_speed_kmh = 0
top_speeds = []

# ===== MAIN LOOP =====
while True:

    # ===== YOLO DETECT =====
    results = model(
        frame,
        conf=0.2,
        imgsz=640,
        verbose=False
    )

    current_speed_kmh = 0

    best_box = None
    best_conf = 0

    # ===== CHOOSE BEST DETECTION =====
    for r in results:

        boxes = r.boxes

        for box in boxes:

            conf = float(box.conf[0])

            if conf > best_conf:
                best_conf = conf
                best_box = box

    # ===== PROCESS BEST BOX =====
    if best_box is not None:

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # ===== SPEED CALC =====
        if prev_center is not None:

            dist_pixel = math.sqrt(
                (cx - prev_center[0]) ** 2 +
                (cy - prev_center[1]) ** 2
            )

            speed_mps = dist_pixel * PIXEL_TO_METER * fps

            current_speed_kmh = speed_mps * 3.6

            # ===== FILTER FAKE SPIKE =====
            if current_speed_kmh < 400:

                # ===== MAX SPEED =====
                if current_speed_kmh > max_speed_kmh:
                    max_speed_kmh = current_speed_kmh

                # ===== TOP 5 =====
                if current_speed_kmh > 10:

                    top_speeds.append(current_speed_kmh)

                    # remove duplicated gần nhau
                    top_speeds = list(set([
                        round(s, 1)
                        for s in top_speeds
                    ]))

                    # sort giảm dần
                    top_speeds = sorted(
                        top_speeds,
                        reverse=True
                    )

                    # giữ top 5
                    top_speeds = top_speeds[:5]

        prev_center = (cx, cy)

        # ===== DRAW BOX =====
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 255, 0),
            -1
        )

        # ===== CONFIDENCE =====
        cv2.putText(
            frame,
            f"Conf: {best_conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    else:
        prev_center = None

    # ===== UI =====
    cv2.putText(
        frame,
        f"Current Speed: {current_speed_kmh:.1f} km/h",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Max Speed: {max_speed_kmh:.1f} km/h",
        (40, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # ===== TOP 5 =====
    for i, s in enumerate(top_speeds):

        cv2.putText(
            frame,
            f"Top {i+1}: {s:.1f} km/h",
            (40, 140 + i * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    # ===== SAVE =====
    out.write(frame)

    # ===== DISPLAY =====
    display = cv2.resize(frame, (1280, 720))

    cv2.imshow(
        "Speed Estimation",
        display
    )

    # ===== EXIT =====
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # ===== NEXT FRAME =====
    ret, frame = cap.read()

    if not ret:
        break

# ===== CLEANUP =====
cap.release()
out.release()

cv2.destroyAllWindows()

# ===== RESULT =====
print("Done:", output_path)

print(f"Max speed: {max_speed_kmh:.2f} km/h")

print("\nTop 5 Speeds:")

for i, s in enumerate(top_speeds):
    print(f"{i+1}. {s:.2f} km/h")