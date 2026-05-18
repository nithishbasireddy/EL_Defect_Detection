import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.el_dataset import ELDataset
from src.models.unet_model import get_model
from src.utils.transforms import get_val_transforms

NUM_CLASSES = 5


def dice_score(pred, target, num_classes=NUM_CLASSES):

    dice_scores = []

    pred = torch.argmax(pred, dim=1)

    for cls in range(num_classes):

        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        if union == 0:
            dice = torch.tensor(1.0)
        else:
            dice = (2 * intersection) / union

        dice_scores.append(dice.item())

    return dice_scores


def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT = "data/raw"

    dataset = ELDataset(
        ROOT,
        split="test_mix",
        transform=get_val_transforms()
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = get_model().to(device)

    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device, weights_only=True))

    model.eval()

    dice_total = np.zeros(NUM_CLASSES)

    with torch.no_grad():

        for imgs, masks in tqdm(loader):

            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)

            dice = dice_score(outputs, masks)

            dice_total += np.array(dice)

    dice_total /= len(loader)

    class_names = [
        "background",
        "dark",
        "cross",
        "crack",
        "busbar"
    ]

    print("\nDice Scores:")
    for i in range(NUM_CLASSES):
        print(f"{class_names[i]}: {dice_total[i]:.4f}")


if __name__ == "__main__":
    evaluate()