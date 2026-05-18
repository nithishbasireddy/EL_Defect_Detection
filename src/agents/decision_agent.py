import cv2
import numpy as np


def compute_dark_percent(cell, mask, ref_mean=None):
    """Compute the dark area percentage for a single cell.

    Three modes (in priority order):
    1. Reference-based: compare against known-good module intensity
    2. Dual-gate (model + intensity): requires BOTH model prediction
       AND low intensity to count as dark (reduces false positives)
    3. Border pixels (outer 10%) are EXCLUDED from dark calculation
       since they are naturally darker due to interconnects.
    """
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- Exclude border pixels (outer 10%) ---
    border_frac = 0.05
    bh = max(1, int(h * border_frac))
    bw = max(1, int(w * border_frac))

    interior_mask = np.zeros((h, w), dtype=bool)
    interior_mask[bh:h-bh, bw:w-bw] = True
    interior_pixels = np.sum(interior_mask)

    if interior_pixels == 0:
        return 0.0

    if ref_mean is not None:
        # --- Reference-based mode ---
        test_mean = np.mean(gray[interior_mask])
        scale = ref_mean / (test_mean + 1e-6)
        normalized = np.clip(gray.astype(np.float32) * scale, 0, 255)

        # Adaptive threshold based on reference brightness
        dark_thresh = 0.55 * ref_mean
        dark_pixels = (normalized < dark_thresh) & interior_mask
        return np.sum(dark_pixels) / interior_pixels * 100

    else:
        # --- Dual-gate mode (model AND intensity) ---
        # Gate 1: Model says it's dark (class 1)
        dark_mask_model = (mask == 1)

        # Gate 2: Actually dark by intensity
        # Use adaptive threshold from cell's own histogram
        gray_resized = cv2.resize(gray, (mask.shape[1], mask.shape[0]))
        cell_median = np.median(gray_resized)

        # Threshold: pixels dimmer than 60% of cell median
        dark_threshold = cell_median * 0.60
        dark_intensity = gray_resized < dark_threshold

        # Resize interior mask to match prediction size
        interior_resized = cv2.resize(
            interior_mask.astype(np.uint8),
            (mask.shape[1], mask.shape[0])
        ).astype(bool)

        # Both gates must agree AND pixel must be interior
        final_dark = dark_mask_model & dark_intensity & interior_resized
        interior_count = np.sum(interior_resized)

        if interior_count == 0:
            return 0.0

        return (np.sum(final_dark) / interior_count) * 100


def decide(cell_data, config):
    """Apply pass/fail rules to a single cell.

    Returns a dict with the verdict and list of reasons.
    """
    reasons = []

    if cell_data["crack_length_mm"] > config["max_crack_length"]:
        reasons.append(
            f"crack length {cell_data['crack_length_mm']:.1f}mm "
            f"exceeds limit {config['max_crack_length']}mm"
        )

    if cell_data["dark_percent"] > config["max_dark_percent"]:
        reasons.append(
            f"dark area {cell_data['dark_percent']:.1f}% "
            f"exceeds limit {config['max_dark_percent']}%"
        )

    verdict = "FAIL" if reasons else "PASS"

    return {
        "verdict": verdict,
        "reasons": reasons,
    }


def decide_all(results, config):
    """Run decision logic on all cells and return enriched results."""
    for r in results:
        decision = decide(r, config)
        r["verdict"] = decision["verdict"]
        r["reasons"] = decision["reasons"]

    return results
