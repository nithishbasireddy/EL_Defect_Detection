import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

def generate_intensity_plot(image_path, save_path):
    print(f"Loading image from {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image. Cannot generate plot.")
        return

    # Proper EL image logic from modules_segmentation.py
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Blur
    ksize_h = max(15, int(h * 0.02) | 1)
    ksize_w = max(15, int(w * 0.02) | 1)
    blur = cv2.GaussianBlur(gray, (ksize_w, ksize_h), 0)
    
    # Compute vertical profile
    col_profile = np.mean(blur, axis=0)
    col_norm = (col_profile - col_profile.min()) / (col_profile.max() - col_profile.min() + 1e-8)
    inverted_col = 1 - col_norm
    
    # Find peaks
    col_peaks, properties = find_peaks(
        inverted_col,
        height=0.25,
        distance=max(10, int(w * 0.03)),
        prominence=0.08
    )
    print(f"Found {len(col_peaks)} peaks.")

    # Plot
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    # Plot profile
    ax.plot(inverted_col, color='#1f77b4', linewidth=2, label="Normalized Intensity\n(Column Projection Profile)")
    
    # Plot peaks
    ax.scatter(col_peaks, inverted_col[col_peaks], color='red', s=100, marker='x', label="Detected Grid Lines (Peaks)", zorder=5)
    
    # Highlight peaks with vertical lines
    for p in col_peaks:
        ax.axvline(x=p, color='red', linestyle='--', alpha=0.3)
        
    ax.set_title("Cell Segmentation: Vertical Intensity Profile & Peak Detection", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Pixel Column (Width)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Inverted Intensity", fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=12, loc='lower center', frameon=True, facecolor='white', framealpha=0.9, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved graph to {save_path}")

if __name__ == "__main__":
    target_img = "data/raw/test_example/NG101070012271_1_Finger Interruptions.jpg"
    if os.path.exists(target_img):
        generate_intensity_plot(target_img, "slide7_intensity_peaks.png")
    else:
        print(f"Could not find {target_img}.")
