import os
import json
from collections import Counter
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

SPLIT_NAME = "test_crack"  

DATA_ROOT = os.path.join(PROJECT_ROOT, "data/raw", SPLIT_NAME)
IMG_DIR = os.path.join(DATA_ROOT, "img")
ANN_DIR = os.path.join(DATA_ROOT, "ann_json")


def compute_class_stats(img_dir, ann_dir):
    image_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    class_counter = Counter()
    images_with_crack = 0

    for img_name in tqdm(image_files):
        json_path = os.path.join(ann_dir, img_name + ".json")

        with open(json_path, "r") as f:
            data = json.load(f)

        classes_in_image = set()

        for obj in data["objects"]:
            class_title = obj["classTitle"]
            class_counter[class_title] += 1
            classes_in_image.add(class_title)

        if "crack" in classes_in_image:
            images_with_crack += 1

    print("---- CLASS COUNTS ----")
    for cls, count in class_counter.items():
        print(f"{cls}: {count}")

    print("\nImages containing crack:", images_with_crack)
    print("Total images:", len(image_files))


if __name__ == "__main__":
    compute_class_stats(IMG_DIR, ANN_DIR)