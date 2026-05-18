import cv2
import numpy as np

def clean_mask_classwise(mask):
    """
    Cleans the predicted mask class by class to prevent mixing classes
    and to apply specific kernels/area thresholds per class.
    
    Classes:
    1: dark
    2: cross
    3: crack
    4: busbar
    """
    clean_mask = np.zeros_like(mask)
    
    for cls in [1, 2, 3, 4]:
        class_mask = (mask == cls).astype(np.uint8)
        
        # Apply class-specific morphology
        if cls == 3:
            # Crack: thinner kernel to preserve thin lines
            kernel = np.ones((2, 2), np.uint8)
            min_area = 30
        else:
            # Others: standard kernel
            kernel = np.ones((3, 3), np.uint8)
            min_area = 50
            
        # Remove morphological noise
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small disconnected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                class_mask[labels == i] = 0
                
        clean_mask[class_mask == 1] = cls
        
    return clean_mask
