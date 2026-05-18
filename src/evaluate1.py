import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Fix import path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.datasets.el_dataset import ELDataset
from src.models.unet_model import get_model
from src.utils.transforms import get_val_transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset
dataset = ELDataset(
    "data/raw",
    split="test_mix",
    transform=get_val_transforms()
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# model
model = get_model().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()


num_classes = 5

class_names = [
    "background",
    "dark",
    "cross",
    "crack",
    "busbar"
]

# global counters
intersection = [0] * num_classes
union = [0] * num_classes


with torch.no_grad():

    for imgs, masks in tqdm(loader):

        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)

        preds = torch.argmax(outputs, dim=1)

        for cls in range(num_classes):

            pred_cls = (preds == cls)
            target_cls = (masks == cls)

            intersection[cls] += (pred_cls & target_cls).sum().item()

            union[cls] += pred_cls.sum().item() + target_cls.sum().item()


print("\nDice Scores:\n")

for cls in range(num_classes):

    if union[cls] == 0:
        dice = 1.0
    else:
        dice = (2 * intersection[cls]) / union[cls]

    print(f"{class_names[cls]}: {dice:.4f}")