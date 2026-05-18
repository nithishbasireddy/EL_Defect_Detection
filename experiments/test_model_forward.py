import os
import sys
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.datasets.el_dataset import ELDataset
from src.utils.transforms import get_train_transforms
from src.models.unet_model import get_model
from src.losses.losses import get_loss

ROOT = os.path.join(PROJECT_ROOT, "data/raw")

dataset = ELDataset(ROOT, split="train", transform=get_train_transforms())

img, mask = dataset[0]

img = img.unsqueeze(0)  # add batch dimension

model = get_model()
loss_fn = get_loss()

model.eval()

with torch.no_grad():
    output = model(img)

print("Model output shape:", output.shape)

loss = loss_fn(output, mask.unsqueeze(0))

print("Loss value:", loss.item())