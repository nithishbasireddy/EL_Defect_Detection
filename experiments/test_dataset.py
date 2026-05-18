import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.datasets.el_dataset import ELDataset 
from src.utils.transforms import get_train_transforms  
ROOT = os.path.join(PROJECT_ROOT, "data/raw")

dataset = ELDataset(ROOT, split="train", transform=get_train_transforms())

print("Dataset size:", len(dataset))

img, mask = dataset[921]

print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
print("Mask unique values:", mask.unique())


height, width = img.shape[1], img.shape[2]

print("Module height (pixels):", height)
print("Module width (pixels):", width)


real_module_width_mm = 2000  
mm_per_pixel = real_module_width_mm / width

print("mm per pixel:", mm_per_pixel)

module_height_mm = height * mm_per_pixel
module_width_mm = width * mm_per_pixel

print("Module height (mm):", module_height_mm)
print("Module width (mm):", module_width_mm)

import random

for i in random.sample(range(len(dataset)), 2):
    _, mask = dataset[i]
    print(f"Sample {i} unique classes:", mask.unique())