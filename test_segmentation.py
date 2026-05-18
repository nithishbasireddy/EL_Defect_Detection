"""Quick test: run pipeline on full module images and report cell counts."""
import os
import sys
import torch
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.full_pipeline import process_image
from src.pipeline.inference import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=5)
model = load_model(model, "checkpoints/best_model.pth", device)

transform = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])

test_dir = "data/raw/test_example"

# Test only larger images (full modules, not single cells)
for fname in sorted(os.listdir(test_dir)):
    if not fname.lower().endswith(('.jpg', '.png')):
        continue
    fpath = os.path.join(test_dir, fname)
    fsize = os.path.getsize(fpath)
    
    # Only test full module images (>100KB)
    if fsize < 100_000:
        continue
    
    img = cv2.imread(fpath)
    h, w = img.shape[:2]
    
    results = process_image(fpath, model, transform, device)
    
    n_cells = len(results)
    total_crack = sum(r["crack_length_mm"] for r in results)
    
    print(f"{fname}: {w}x{h} -> {n_cells} cells, total crack={total_crack:.1f}mm")
    
    # Show first 3 cells' crack info
    for r in results[:3]:
        print(f"  Cell {r['cell_id']}: crack={r['crack_length_mm']:.1f}mm, severity={r['severity']}")
