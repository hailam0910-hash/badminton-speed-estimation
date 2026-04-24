import cv2
import os

video_path = "videos/video1.mp4"
save_dir = "dataset/images_raw"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
save_id = 0

# lấy 1 frame mỗi 10 frame
interval = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % interval == 0:
        cv2.imwrite(f"{save_dir}/frame_{save_id:04d}.jpg", frame)
        save_id += 1

    frame_id += 1

cap.release()
print("Done, saved frames:", save_id)