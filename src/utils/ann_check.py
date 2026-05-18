import os
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

mask_path = os.path.join(PROJECT_ROOT, "data/raw/train/ann/0007.png")

print("Mask path:", mask_path)
print("Exists:", os.path.exists(mask_path))

mask = cv2.imread(mask_path, 0)

print("Mask shape:", None if mask is None else mask.shape)
print("Unique values:", None if mask is None else np.unique(mask))