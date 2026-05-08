import os

image_dir = "dataset/new_images_video2"
label_dir = "dataset/new_labels_video2"

prefix = "video2"

image_files = sorted(os.listdir(image_dir))

for i, img_name in enumerate(image_files):
    if not img_name.endswith(".jpg"):
        continue

    new_name = f"{prefix}_{i:04d}.jpg"
    old_img_path = os.path.join(image_dir, img_name)
    new_img_path = os.path.join(image_dir, new_name)

    os.rename(old_img_path, new_img_path)

    # rename label tương ứng
    label_name = img_name.replace(".jpg", ".txt")
    old_label_path = os.path.join(label_dir, label_name)

    new_label_name = new_name.replace(".jpg", ".txt")
    new_label_path = os.path.join(label_dir, new_label_name)

    if os.path.exists(old_label_path):
        os.rename(old_label_path, new_label_path)

print("Rename done")