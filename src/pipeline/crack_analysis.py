import cv2
import numpy as np
from skimage.morphology import skeletonize


# -----------------------
# EXTRACT CRACK MASK
# -----------------------
def get_crack_mask(mask):
    # class 3 = crack
    crack = (mask == 3).astype(np.uint8)
    return crack


# -----------------------
# REMOVE NOISE
# -----------------------
def clean_crack_mask(crack_mask):
    # Use thinner kernel to preserve thin lines
    kernel = np.ones((2,2), np.uint8)
    # The mask should already be pre-cleaned upstream, but this is an extra safety layer
    cleaned = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel)
    return cleaned


# -----------------------
# SKELETONIZE
# -----------------------
def get_skeleton(crack_mask):
    skeleton = skeletonize(crack_mask > 0)
    return skeleton.astype(np.uint8)


# -----------------------
# LENGTH CALCULATION
# -----------------------
def compute_length(skeleton):
    return np.sum(skeleton)


# -----------------------
# PIXEL to MM
# -----------------------
def pixel_to_mm(length_pixels):
    mm_per_pixel = 156 / 512  
    return length_pixels * mm_per_pixel


# -----------------------
# SEVERITY
# -----------------------
def classify_severity(length_mm):
    if length_mm == 0:
        return "none"
    elif length_mm < 10:
        return "minor"
    elif length_mm < 20:
        return "moderate"
    else:
        return "severe"


# -----------------------
# GRID-LINE FALSE POSITIVE FILTERS
# -----------------------

def filter_edge_touching(skeleton, border_px=5):
    """Remove skeleton pixels that sit at cell edges.
    
    Grid bar fragments always appear at the border of a mis-cropped cell.
    Real cracks typically originate mid-cell, not at the very edge.
    
    If >85% of a connected component's pixels are within border_px
    of the cell edge, it is a grid bar artifact, not a real crack.
    Diagonal cracks naturally touch edges but have significant mid-cell presence.
    """
    h, w = skeleton.shape
    filtered = skeleton.copy()

    # Create border mask
    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_mask[:border_px, :] = 1      # top
    border_mask[h-border_px:, :] = 1    # bottom
    border_mask[:, :border_px] = 1      # left
    border_mask[:, w-border_px:] = 1    # right

    # Label connected components in skeleton
    num_labels, labels = cv2.connectedComponents(skeleton)

    for label_id in range(1, num_labels):
        component = (labels == label_id).astype(np.uint8)
        total_px = np.sum(component)

        if total_px == 0:
            continue

        border_px_count = np.sum(component & border_mask)
        border_ratio = border_px_count / total_px

        # If >85% of this crack sits at the edge, it's a grid bar
        # (relaxed from 70% to preserve diagonal cracks)
        if border_ratio > 0.85:
            filtered[labels == label_id] = 0

    return filtered


def filter_straight_lines(skeleton, straightness_threshold=0.90):
    """Remove skeleton components that are perfectly straight.
    
    Grid bars are perfectly horizontal or vertical lines.
    Real cracks are irregular, jagged, branching.
    
    We check: if the bounding box of a component is very thin
    in one dimension (width or height < 5px), it's a straight line.
    """
    h, w = skeleton.shape
    filtered = skeleton.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton)

    for label_id in range(1, num_labels):
        comp_w = stats[label_id, cv2.CC_STAT_WIDTH]
        comp_h = stats[label_id, cv2.CC_STAT_HEIGHT]
        comp_area = stats[label_id, cv2.CC_STAT_AREA]

        if comp_area < 5:
            # Too tiny to judge
            filtered[labels == label_id] = 0
            continue

        # Check if component spans nearly the full cell width or height
        spans_full_width = comp_w > w * 0.75
        spans_full_height = comp_h > h * 0.75

        # Check if it's very thin (straight line)
        is_thin_horizontal = comp_h < max(6, h * 0.03) and comp_w > w * 0.3
        is_thin_vertical = comp_w < max(6, w * 0.03) and comp_h > h * 0.3

        if is_thin_horizontal or is_thin_vertical:
            filtered[labels == label_id] = 0
        elif spans_full_width and comp_h < h * 0.05:
            filtered[labels == label_id] = 0
        elif spans_full_height and comp_w < w * 0.05:
            filtered[labels == label_id] = 0

    return filtered


# -----------------------
# FULL PIPELINE
# -----------------------
def analyze_crack(mask):

    crack = get_crack_mask(mask)
    crack = clean_crack_mask(crack)

    skeleton = get_skeleton(crack)

    # --- CRITICAL: Filter false positives ---
    skeleton = filter_edge_touching(skeleton)
    skeleton = filter_straight_lines(skeleton)

    length_px = compute_length(skeleton)
    
    # Ignore tiny cracks (noise)
    if length_px < 5:
        length_px = 0
        
    length_mm = pixel_to_mm(length_px)

    # Physical sanity cap: a 156mm cell can't have a crack longer
    # than its diagonal (~220mm). If we see more, it's grid artifacts.
    MAX_POSSIBLE_CRACK_MM = 22000.0
    if length_mm > MAX_POSSIBLE_CRACK_MM:
        length_mm = 0.0
        length_px = 0

    severity = classify_severity(length_mm)

    # Get individual cracks using connected components
    num_labels, labels = cv2.connectedComponents(skeleton)
    individual_cracks_mm = []
    
    for label_id in range(1, num_labels):
        comp = (labels == label_id).astype(np.uint8)
        px_len = compute_length(comp)
        
        # Ignore tiny individual noise lines
        if px_len >= 5:
            mm_len = pixel_to_mm(px_len)
            individual_cracks_mm.append(round(mm_len, 2))
            
    # Sort from longest to shortest
    individual_cracks_mm.sort(reverse=True)

    return {
        "length_px": length_px,
        "length_mm": length_mm,
        "severity": severity,
        "individual_cracks_mm": individual_cracks_mm,
        "skeleton": skeleton
    }