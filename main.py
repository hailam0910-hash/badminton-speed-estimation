import cv2
import numpy as np

video_path = "test.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không mở được video")
    exit()

print("Mở video thành công")
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = cap.read()

if ret:
    cv2.imwrite("first_frame.jpg", frame)
    print("Đã lưu frame đầu tiên: first_frame.jpg")
else:
    print("Không đọc được frame")

cap.release()