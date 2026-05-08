import cv2
import os
import glob
import shutil
import numpy as np

input_dir = "dataset/images_raw_video2"
output_dir = "dataset/selected_different_video2"

os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

prev_gray = None
saved_count = 0

# chỉnh số này
DIFF_THRESHOLD = 5.0

for img_path in image_paths:
    img = cv2.imread(img_path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (320, 180))

    if prev_gray is None:
        shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))
        prev_gray = gray
        saved_count += 1
        continue

    diff = cv2.absdiff(gray, prev_gray)
    score = np.mean(diff)

    if score > DIFF_THRESHOLD:
        shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))
        saved_count += 1
        prev_gray = gray

print("Total input:", len(image_paths))
print("Selected:", saved_count)
print("Saved to:", output_dir)