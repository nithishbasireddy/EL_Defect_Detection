import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class ELDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(root_dir, split, "img")
        self.mask_dir = os.path.join(root_dir, split, "ann")

        self.image_files = [
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png").replace(".jpeg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        if mask is None:
            raise ValueError(f"Mask not found for {img_name}")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()  # ToTensorV2 gives uint8; CrossEntropyLoss needs Long
        else:
            # fallback if no transform is provided
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        if getattr(self, "return_name", False):
            return image, mask, img_name

        return image, mask