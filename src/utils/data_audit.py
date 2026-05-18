import os
import json
import cv2
import numpy as np
from tqdm import tqdm

#
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

SPLIT_NAME = "test_mix"  # val/test_mix/test_crack 

DATA_ROOT = os.path.join(PROJECT_ROOT, "data/raw", SPLIT_NAME)
IMG_DIR = os.path.join(DATA_ROOT, "img")
ANN_DIR = os.path.join(DATA_ROOT, "ann_json")

def audit_dataset(img_dir, ann_dir):
    image_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Total images found: {len(image_files)}")

    missing_json = 0
    dimension_mismatch = 0
    empty_masks = 0

    for img_name in tqdm(image_files):
        img_path = os.path.join(img_dir, img_name)
        json_name = img_name + ".json"
        json_path = os.path.join(ann_dir, json_name)

        if not os.path.exists(json_path):
            missing_json += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Corrupted image: {img_name}")
            continue

        h, w = img.shape[:2]

        with open(json_path, "r") as f:
            data = json.load(f)

        json_h = data["size"]["height"]
        json_w = data["size"]["width"]

        if h != json_h or w != json_w:
            dimension_mismatch += 1

        if len(data["objects"]) == 0:
            empty_masks += 1

    print("---- AUDIT RESULTS ----")
    print(f"Missing JSON files: {missing_json}")
    print(f"Dimension mismatch: {dimension_mismatch}")
    print(f"Images with no objects: {empty_masks}")

if __name__ == "__main__":
    audit_dataset(IMG_DIR, ANN_DIR)    