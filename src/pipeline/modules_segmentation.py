import cv2
import numpy as np
from scipy.signal import find_peaks


# -----------------------
# ADAPTIVE CONTRAST ENHANCEMENT
# -----------------------
def enhance_contrast(gray):
    """Apply CLAHE to make grid lines visible in dark/uneven images.
    
    CLAHE (Contrast Limited Adaptive Histogram Equalization) boosts
    local contrast without washing out bright areas. Critical for
    extremely dark modules where grid lines are invisible in raw pixels.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# -----------------------
# REMOVE BLACK BORDERS (ADAPTIVE)
# -----------------------
def remove_borders(img):
    """Remove black borders around the module.
    
    Uses Otsu's thresholding instead of fixed mean-std formula,
    which handles both very dark and very bright images.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)

    # Otsu finds optimal threshold automatically — works on any brightness
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find bounding box of the non-zero region
    coords = cv2.findNonZero(binary)

    if coords is None:
        # Entire image is dark — fall back to original approach
        mean_val = np.mean(blur)
        std_val = np.std(blur)
        threshold = max(mean_val - 1.5 * std_val, np.percentile(blur, 10))

        h, w = gray.shape
        top = next((i for i in range(h) if np.mean(blur[i, :]) > threshold), 0)
        bottom = next((i for i in range(h-1, -1, -1) if np.mean(blur[i, :]) > threshold), h-1)
        left = next((j for j in range(w) if np.mean(blur[:, j]) > threshold), 0)
        right = next((j for j in range(w-1, -1, -1) if np.mean(blur[:, j]) > threshold), w-1)
    else:
        x, y, bw, bh = cv2.boundingRect(coords)
        top, bottom = y, y + bh
        left, right = x, x + bw

    margin = 5
    h, w = gray.shape

    cropped = img[
        max(0, top - margin):min(h, bottom + margin),
        max(0, left - margin):min(w, right + margin)
    ]

    # Safety: don't return tiny crops (border removal ate the image)
    if cropped.shape[0] < img.shape[0] * 0.3 or cropped.shape[1] < img.shape[1] * 0.3:
        return img

    return cropped


# -----------------------
# SINGLE CELL CHECK
# -----------------------
def is_single_cell(img):
    h, w = img.shape[:2]
    aspect = max(h, w) / max(min(h, w), 1)
    area = h * w

    if aspect < 1.8 and area < 500_000:
        return True

    return False


# -----------------------
# FIND DOMINANT SPACING
# -----------------------
def find_dominant_spacing(spacings):
    if len(spacings) == 0:
        return 0

    if len(spacings) <= 2:
        return np.median(spacings)

    sorted_spacings = np.sort(spacings)
    min_s, max_s = sorted_spacings[0], sorted_spacings[-1]

    if max_s - min_s < max_s * 0.3:
        return np.median(spacings)

    gaps = np.diff(sorted_spacings)
    largest_gap_idx = np.argmax(gaps)

    if gaps[largest_gap_idx] > (max_s - min_s) * 0.2:
        small_group = sorted_spacings[:largest_gap_idx + 1]
        large_group = sorted_spacings[largest_gap_idx + 1:]

        if len(small_group) > 0 and len(large_group) > 0:
            small_mean = np.mean(small_group)
            large_mean = np.mean(large_group)

            # Detect busbar vs cell spacing
            if 1.4 < large_mean / small_mean < 3.5:
                return large_mean

    return np.median(spacings)


# -----------------------
# PERIODICITY FILTER
# -----------------------
def filter_peaks_by_periodicity(peaks, total_length):
    if len(peaks) < 2:
        return peaks

    peaks = np.sort(peaks)

    for _ in range(5):
        if len(peaks) < 2:
            break

        spacings = np.diff(peaks)
        dominant_spacing = find_dominant_spacing(spacings)

        if dominant_spacing < total_length * 0.03:
            break

        tolerance = 0.35
        lo = dominant_spacing * (1 - tolerance)
        hi = dominant_spacing * (1 + tolerance)

        filtered = [peaks[0]]

        for i in range(1, len(peaks)):
            gap = peaks[i] - filtered[-1]

            if lo <= gap <= hi:
                filtered.append(peaks[i])
            elif gap < lo:
                continue
            else:
                filtered.append(peaks[i])

        new_peaks = np.array(filtered)

        if len(new_peaks) == len(peaks):
            break

        peaks = new_peaks

    return peaks


# -----------------------
# FILTER BUSBAR PEAKS
# -----------------------
def filter_busbar_peaks(peaks, total_length):
    """Remove busbar peaks that are much closer together than cell boundaries.
    
    Busbars create narrow dark lines WITHIN a cell. Their spacing is
    ~1/3 to 1/2 of the real cell spacing. We detect the dominant
    (largest) spacing group and remove peaks that create smaller gaps.
    """
    if len(peaks) < 3:
        return peaks

    spacings = np.diff(peaks)
    if len(spacings) < 2:
        return peaks

    median_spacing = np.median(spacings)

    # Keep only peaks that create gaps >= 60% of the median spacing
    # This removes busbar peaks (which create ~30-50% of cell spacing)
    min_allowed = median_spacing * 0.55

    filtered = [peaks[0]]
    for i in range(1, len(peaks)):
        gap = peaks[i] - filtered[-1]
        if gap >= min_allowed:
            filtered.append(peaks[i])

    return np.array(filtered)


# -----------------------
# GRID VALIDATION (CRITICAL)
# -----------------------
def validate_grid(row_peaks, col_peaks):
    if len(row_peaks) < 2 or len(col_peaks) < 2:
        return False

    row_spacings = np.diff(row_peaks)
    col_spacings = np.diff(col_peaks)

    if np.std(row_spacings) > 0.40 * np.mean(row_spacings):
        return False

    if np.std(col_spacings) > 0.40 * np.mean(col_spacings):
        return False

    return True


# -----------------------
# DETECT GRID (IMPROVED)
# -----------------------
def detect_grid(gray):
    """Detect grid lines using projection profiles on CLAHE-enhanced image.
    
    Key improvement: CLAHE preprocessing makes grid visible even in
    extremely dark modules where raw pixel values show no structure.
    """
    h, w = gray.shape

    # --- CRITICAL FIX: Enhance contrast BEFORE projection ---
    enhanced = enhance_contrast(gray)

    ksize_h = max(15, int(h * 0.02) | 1)
    ksize_w = max(15, int(w * 0.02) | 1)

    blur = cv2.GaussianBlur(enhanced, (ksize_w, ksize_h), 0)

    # Adaptive peak parameters based on image brightness
    img_brightness = np.mean(gray)
    if img_brightness < 60:
        # Very dark image — lower thresholds to catch faint grid
        peak_height = 0.15
        peak_prominence = 0.05
    elif img_brightness < 120:
        # Medium brightness
        peak_height = 0.20
        peak_prominence = 0.06
    else:
        # Bright image — standard thresholds
        peak_height = 0.25
        peak_prominence = 0.08

    # Vertical grid lines (column boundaries)
    col_profile = np.mean(blur, axis=0)
    col_norm = (col_profile - col_profile.min()) / (col_profile.max() - col_profile.min() + 1e-8)
    inverted_col = 1 - col_norm

    col_peaks, _ = find_peaks(
        inverted_col,
        height=peak_height,
        distance=max(10, int(w * 0.03)),
        prominence=peak_prominence
    )

    # Horizontal grid lines (row boundaries)
    row_profile = np.mean(blur, axis=1)
    row_norm = (row_profile - row_profile.min()) / (row_profile.max() - row_profile.min() + 1e-8)
    inverted_row = 1 - row_norm

    row_peaks, _ = find_peaks(
        inverted_row,
        height=peak_height,
        distance=max(10, int(h * 0.05)),
        prominence=peak_prominence
    )

    col_peaks = filter_peaks_by_periodicity(col_peaks, w)
    row_peaks = filter_peaks_by_periodicity(row_peaks, h)

    # --- CRITICAL: Remove busbar peaks ---
    col_peaks = filter_busbar_peaks(col_peaks, w)
    row_peaks = filter_busbar_peaks(row_peaks, h)

    return row_peaks, col_peaks


# -----------------------
# MAIN FUNCTION
# -----------------------
def segment_cells(image):

    cropped = remove_borders(image)

    if is_single_cell(cropped):
        return [cropped]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    row_peaks, col_peaks = detect_grid(gray)

    # VALIDATION STEP (MOST IMPORTANT)
    if not validate_grid(row_peaks, col_peaks):
        return [cropped]

    h, w = gray.shape

    row_boundaries = np.concatenate(([0], row_peaks, [h]))
    col_boundaries = np.concatenate(([0], col_peaks, [w]))

    cells = []

    min_cell_h = h * 0.03
    min_cell_w = w * 0.03

    for i in range(len(row_boundaries)-1):
        for j in range(len(col_boundaries)-1):
            y1, y2 = int(row_boundaries[i]), int(row_boundaries[i+1])
            x1, x2 = int(col_boundaries[j]), int(col_boundaries[j+1])

            cell = cropped[y1:y2, x1:x2]

            if cell.shape[0] > min_cell_h and cell.shape[1] > min_cell_w:
                cells.append(cell)

    if len(cells) == 0:
        return [cropped]

    # SANITY CHECK: standard modules have 36-72 cells
    # If we get way too many, busbar filtering failed
    if len(cells) > 100:
        return [cropped]

    return cells