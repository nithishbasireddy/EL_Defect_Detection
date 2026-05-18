import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_dark_plot():
    # Use an image to crop a realistic background
    img_path = "data/raw/test_example/NG101070012271_1_Finger Interruptions.jpg"
    if os.path.exists(img_path):
        img_full = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Extract a decent size patch 512x512
        cell = img_full[100:612, 100:612]
        if cell.shape != (512, 512):
            cell = np.ones((512, 512), dtype=np.uint8) * 150
    else:
        cell = np.ones((512, 512), dtype=np.uint8) * 150
        
    # Baseline Reference Mean (Perfect lighting)
    ref_mean = 160.0
    
    # 1. We create a simulated poorly lit test cell (darker overall due to camera/flash variance)
    # Using cv2.convertScaleAbs to darken it
    poorly_lit_cell = cv2.convertScaleAbs(cell, alpha=0.55, beta=10)
    
    # 2. Inject a genuine "Dark Area" Defect into the cell (bottom right smudge)
    true_defect_cell = poorly_lit_cell.copy()
    cv2.circle(true_defect_cell, (380, 380), 60, (20, 20, 20), -1)
    true_defect_cell = cv2.GaussianBlur(true_defect_cell, (21, 21), 0) # soften
    
    test_mean = np.mean(true_defect_cell)
    
    # Scenario A: WITHOUT REFERENCE (Naive/Model-based failure case)
    # The absolute threshold is 60% of what normally is reference mean
    # Because the whole image is poorly lit, a naive approach flags too much
    naive_dark_threshold = 0.6 * ref_mean
    naive_detected = true_defect_cell < naive_dark_threshold
    
    # Scenario B: WITH REFERENCE (Your robust app.py logic)
    # scale = ref_mean / test_mean
    scale = ref_mean / (test_mean + 1e-6)
    normalized_cell = np.clip(true_defect_cell * scale, 0, 255).astype(np.uint8)
    
    # Now detect dark on the normalized version
    robust_detected = normalized_cell < (0.6 * ref_mean)
    
    # Let's plot it beautifully
    try: plt.style.use('seaborn-v0_8-darkgrid')
    except: pass
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), dpi=300)
    
    axs[0].imshow(true_defect_cell, cmap="gray", vmin=0, vmax=255)
    axs[0].set_title("Input Sub-Cell\n(Poor Lighting + Real Defect)", fontsize=14, fontweight='bold', pad=10)
    axs[0].axis("off")
    
    # Red-tint the naive detection overlay
    naive_overlay = cv2.cvtColor(true_defect_cell, cv2.COLOR_GRAY2RGB)
    naive_overlay[naive_detected] = [255, 50, 50]
    axs[1].imshow(naive_overlay)
    axs[1].set_title("❌ WITHOUT REFERENCE\nFails: Entire cell is flagged dark", fontsize=14, fontweight='bold', pad=10, color='#d62728')
    axs[1].axis("off")
    
    axs[2].imshow(normalized_cell, cmap="gray", vmin=0, vmax=255)
    axs[2].set_title("Normalization Step\n(scale = ref_mean / test_mean)", fontsize=14, fontweight='bold', pad=10, color='#1f77b4')
    axs[2].axis("off")
    
    # Green-tint the robust detection overlay
    robust_overlay = cv2.cvtColor(normalized_cell, cv2.COLOR_GRAY2RGB)
    robust_overlay[robust_detected] = [50, 255, 50]
    axs[3].imshow(robust_overlay)
    axs[3].set_title("✅ WITH REFERENCE\nSuccess: Only true defect flagged", fontsize=14, fontweight='bold', pad=10, color='#2ca02c')
    axs[3].axis("off")
    
    plt.tight_layout()
    save_path = "slide8_dark_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Dark comparison diagram saved to {save_path}")

if __name__ == "__main__":
    generate_dark_plot()
