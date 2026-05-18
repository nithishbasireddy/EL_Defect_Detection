import numpy as np
import cv2


def overlay_mask(image, mask):
    """
    Overlay segmentation mask on image with colors
    """

    h, w = image.shape[:2]

    # resize mask to match image
    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    color_map = {
        1: (255, 0, 0),    # dark → blue
        2: (0, 255, 255),  # cross → yellow
        3: (0, 0, 255),    # crack → red
        4: (0, 255, 0)     # busbar → green
    }

    overlay = image.copy()

    for cls, color in color_map.items():
        overlay[mask_resized == cls] = color

    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    return blended