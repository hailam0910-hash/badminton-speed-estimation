import cv2
import os

# ===== CONFIG =====
video_path = "videos/IMG_9806.MOV"
save_dir = "dataset/images_raw_video2"
prefix = "video2"   # 👉 đổi theo từng video (video1, video2, v3...)

os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không mở được video")
    exit()

frame_id = 0
save_id = 0

# lấy 1 frame mỗi N frame
interval = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % interval == 0:
        filename = f"{prefix}_{save_id:04d}.jpg"
        filepath = os.path.join(save_dir, filename)

        cv2.imwrite(filepath, frame)
        save_id += 1

    frame_id += 1

cap.release()

print(f"Done, saved frames: {save_id}")
print(f"Saved to: {save_dir}")