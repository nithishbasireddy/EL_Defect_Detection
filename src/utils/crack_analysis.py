import numpy as np
import cv2


# constant (based on your dataset)
MM_PER_PIXEL = 156 / 400


def analyze_cracks(pred_mask):
    """
    Input:
        pred_mask (H, W) numpy array with class labels

    Output:
        dict with crack statistics
    """

    # extract masks
    crack_mask = (pred_mask == 3).astype(np.uint8)
    busbar_mask = (pred_mask == 4).astype(np.uint8)

    # ---- Step 1: clean noise ----
    crack_mask = cv2.medianBlur(crack_mask, 3)

    # remove very small regions
    min_pixels = 20
    num_labels, labels = cv2.connectedComponents(crack_mask)

    cleaned_mask = np.zeros_like(crack_mask)

    for i in range(1, num_labels):
        component = (labels == i).astype(np.uint8)

        if np.sum(component) >= min_pixels:
            cleaned_mask += component

    crack_mask = cleaned_mask

    # ---- Step 2: connect broken cracks ----
    kernel = np.ones((3, 3), np.uint8)

    # ---- Step 3: connected components again ----
    num_labels, labels = cv2.connectedComponents(crack_mask)

    crack_info = []

    for i in range(1, num_labels):

        component = (labels == i).astype(np.uint8)

        # connect broken bits within the single component
        kernel = np.ones((3, 3), np.uint8)
        component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel)

        # convert to proper binary image (0,255)
        binary = (component * 255).astype("uint8")

        skeleton = cv2.ximgproc.thinning(binary)

        # convert back to 0/1
        skeleton = (skeleton > 0).astype("uint8")

        length_pixels = np.sum(skeleton)
        length_mm = length_pixels * MM_PER_PIXEL
        print("Component pixels:", np.sum(component))
        print("Skeleton pixels:", np.sum(skeleton))

        # check busbar intersection
        intersects_busbar = np.any((component == 1) & (busbar_mask == 1))

        # severity rules
        if length_mm < 10:
            severity = "minor"
        elif 10 <= length_mm <= 20:
            severity = "moderate"
        else:
            severity = "severe"

        # upgrade if intersects busbar
        if intersects_busbar:
            if severity == "minor":
                severity = "moderate"
            elif severity == "moderate":
                severity = "severe"

        crack_info.append({
            "length_mm": round(length_mm, 2),
            "severity": severity,
            "busbar_intersection": intersects_busbar
        })

    # ---- summary ----
    total_cracks = len(crack_info)

    max_length = max([c["length_mm"] for c in crack_info], default=0)

    return {
        "total_cracks": total_cracks,
        "max_crack_length_mm": round(max_length, 2),
        "cracks": crack_info
    }