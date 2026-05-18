import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.datasets.el_dataset import ELDataset
from src.models.unet_model import get_model
from src.utils.transforms import get_val_transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ELDataset(
    "data/raw",
    split="test_mix",
    transform=get_val_transforms()
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = get_model().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

for imgs, masks in loader:

    imgs = imgs.to(device)

    with torch.no_grad():
        outputs = model(imgs)

    preds = torch.argmax(outputs, dim=1)

    img = imgs.cpu()[0].permute(1,2,0)
    mask = masks[0]
    pred = preds.cpu()[0]

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(mask)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(pred)
    plt.axis("off")

    plt.show()

    break