import cv2
import numpy as np

video_path = "videos/video1.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không mở được video")
    exit()

# ==== PARAM (tune riêng cho video của bạn) ====
MIN_AREA = 5
MAX_AREA = 120
THRESH = 180
MAX_DIST = 150

prev_center = None

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/output.mp4', fourcc, 30,
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # giảm noise nhưng vẫn giữ tốc độ
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # detect vùng sáng
    _, thresh = cv2.threshold(blur, THRESH, 255, cv2.THRESH_BINARY)

    # MORPH để giảm nhiễu đèn nhỏ
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    min_dist = 999999

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # lọc shape (cầu gần tròn)
        ratio = w / h if h != 0 else 0
        if ratio < 0.5 or ratio > 2:
            continue

        cx = x + w // 2
        cy = y + h // 2

        if prev_center is not None:
            dist = np.sqrt((cx - prev_center[0])**2 + (cy - prev_center[1])**2)

            if dist < min_dist and dist < MAX_DIST:
                min_dist = dist
                best = (x, y, w, h, cx, cy)
        else:
            best = (x, y, w, h, cx, cy)

    # ==== DRAW ====
    if best:
        x, y, w, h, cx, cy = best
        prev_center = (cx, cy)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
    else:
        prev_center = None

    out.write(frame)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done!")