import torch
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.pipeline.full_pipeline import process_image
from src.pipeline.inference import load_model

# -----------------------
# DEVICE
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# MODEL
# -----------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    classes=5
)

model = load_model(model, "checkpoints/best_model.pth", device)

# -----------------------
# TRANSFORM
# -----------------------
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])

# -----------------------
# IMAGE PATH
# -----------------------
image_path = "data/raw/test_example/0017.jpg"  

# -----------------------
# RUN PIPELINE
# -----------------------
results = process_image(image_path, model, transform, device)

print(image_path)
print("Total cells detected:", len(results))

for r in results[:10]:
    print(
        f"Cell {r['cell_id']} → Length: {r['crack_length_mm']:.2f} mm → {r['severity']}"
    )