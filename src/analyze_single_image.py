import os
import sys
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.models.unet_model import get_model
from src.utils.transforms import get_val_transforms
from src.utils.crack_analysis import analyze_cracks

def main(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = get_model().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.eval()

    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Transform
    transform = get_val_transforms()
    # Albumentations can process just the image
    augmented = transform(image=image_rgb)
    img_tensor = augmented["image"].unsqueeze(0).to(device) # Add batch dimension

    print(f"\n[INFO] Analyzing: {os.path.basename(image_path)}")

    with torch.no_grad():
        outputs = model(img_tensor)

    preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]

    # Analysis
    result = analyze_cracks(preds)

    print("\n===== CRACK ANALYSIS =====")
    print(f"Total Cracks: {result['total_cracks']}")
    if result['total_cracks'] > 0:
        print(f"Max Crack Length: {result['max_crack_length_mm']:.2f} mm")

    for i, crack in enumerate(result["cracks"], 1):
        print(f"\nCrack {i}:")
        print(f"Length: {crack['length_mm']:.2f} mm")
        print(f"Severity: {crack['severity']}")
        print(f"Busbar Intersection: {crack['busbar_intersection']}")

    # Visualization
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Crack")
    plt.imshow(preds)
    plt.axis("off")
    
    save_path = f"prediction_{os.path.basename(image_path)}"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nSaved visualization to {save_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a single image for cracks.")
    parser.add_argument("image_path", type=str, help="Path to the image to analyze")
    args = parser.parse_args()
    
    main(args.image_path)
