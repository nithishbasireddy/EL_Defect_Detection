import cv2
from .modules_segmentation import segment_cells
from .inference import predict_cell
from .crack_analysis import analyze_crack
from .mask_cleaning import clean_mask_classwise


def process_image(image_path, model, transform, device):
    img = cv2.imread(image_path)

    cells = segment_cells(img)

    results = []

    for idx, cell in enumerate(cells):
        # Convert BGR→RGB to match training preprocessing
        cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
        raw_mask = predict_cell(model, cell_rgb, transform, device)
        clean_mask = clean_mask_classwise(raw_mask)

        crack_info = analyze_crack(clean_mask)

        results.append({
            "cell_id": idx,
            "cell": cell,
            "mask": clean_mask,  # Return cleaned mask
            "crack_length_mm": crack_info["length_mm"],
            "individual_cracks_mm": crack_info["individual_cracks_mm"],
            "severity": crack_info["severity"]
        })

    return resultsid": idx,
            "cell": cell,
            "mask": clean_mask,  # Return cleaned mask
            "crack_length_mm": crack_info["length_mm"],
            "individual_cracks_mm": crack_info["individual_cracks_mm"],
            "severity": crack_info["severity"]
        })

    return results