import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.datasets.el_dataset import ELDataset
from src.models.unet_model import get_model
from src.utils.transforms import get_val_transforms
from src.utils.crack_analysis import analyze_cracks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ELDataset(
    "data/raw",
    split="test_crack",
    transform=get_val_transforms()
)

dataset.return_name = True

loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = get_model().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()


for imgs, masks, img_names in loader:
    
    if img_names[0] != "1625.jpg":
        continue

    print(f"\n[INFO] Currently Analyzing: {img_names[0]}")
    imgs = imgs.to(device)

    with torch.no_grad():
        outputs = model(imgs)

    preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]

    result = analyze_cracks(preds)

    print("\n===== CRACK ANALYSIS =====")
    print(f"Total Cracks: {result['total_cracks']}")
    print(f"Max Crack Length: {result['max_crack_length_mm']} mm")

    for i, crack in enumerate(result["cracks"], 1):
        print(f"\nCrack {i}:")
        print(f"Length: {crack['length_mm']} mm")
        print(f"Severity: {crack['severity']}")
        print(f"Busbar Intersection: {crack['busbar_intersection']}")

    break