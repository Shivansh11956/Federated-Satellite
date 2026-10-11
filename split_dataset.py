import os
import shutil
import random

source = "EuroSAT"
dest1 = "dataset_part1"
dest2 = "dataset_part2"

os.makedirs(dest1, exist_ok=True)
os.makedirs(dest2, exist_ok=True)

for class_name in os.listdir(source):
    class_path = os.path.join(source, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split = int(0.5 * len(images))

    part1 = images[:split]
    part2 = images[split:]

    os.makedirs(os.path.join(dest1, class_name), exist_ok=True)
    os.makedirs(os.path.join(dest2, class_name), exist_ok=True)

    for img in part1:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(dest1, class_name, img)
        )

    for img in part2:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(dest2, class_name, img)
        )

print("Dataset split complete!")