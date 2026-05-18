"""Quick sanity check with verbose error reporting."""
import os, sys, torch, traceback

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from torch.utils.data import DataLoader
from src.datasets.el_dataset import ELDataset
from src.utils.transforms import get_train_transforms
from src.models.unet_model import get_model
from src.losses.losses import loss_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset = ELDataset("data/raw", split="train", transform=get_train_transforms())
loader = DataLoader(dataset, batch_size=4, shuffle=True)

images, masks = next(iter(loader))
print(f"images.shape: {images.shape}")
print(f"masks.shape:  {masks.shape}")
print(f"masks unique: {masks.unique()}")
print(f"masks dtype:  {masks.dtype}")

images = images.to(device)
masks = masks.to(device)

model = get_model().to(device)
model.eval()

with torch.no_grad():
    preds = model(images)

print(f"preds.shape:  {preds.shape}")

try:
    loss = loss_fn(preds, masks)
    print(f"loss value:   {loss.item()}")
    print("\nAll checks passed!")
except Exception as e:
    print(f"\nLoss error: {e}")
    traceback