import cv2
import numpy as np

from .decision_agent import compute_dark_percent, decide_all
from .explanation_agent import generate_report


def run_inspection(results, config, ref_mean=None):
    """Orchestrate the full inspection flow across all agents.

    Pipeline:
        1. Compute dark_percent for each cell (Decision Agent)
        2. Apply pass/fail rules to each cell (Decision Agent)
        3. Aggregate module-level statistics

    Parameters:
        results: list of dicts from process_image()
        config: dict with threshold settings
        ref_mean: float, mean intensity of reference image (or None)

    Returns:
        dict with enriched results and summary statistics
    """
    # --- Step 1: Compute dark_percent for each cell ---
    for r in results:
        r["dark_percent"] = compute_dark_percent(
            r["cell"], r["mask"], ref_mean
        )

    # --- Step 2: Apply decision rules ---
    results = decide_all(results, config)

    # --- Step 3: Aggregate statistics ---
    total = len(results)
    failed = [r for r in results if r["verdict"] == "FAIL"]

    total_crack = sum(r["crack_length_mm"] for r in results)

    summary = {
        "total_cells": total,
        "defective_cells": len(failed),
        "total_crack_length": total_crack,
        "module_verdict": "FAIL" if failed else "PASS",
    }

    return {
        "results": results,
        "summary": summary,
        "config": config,
    }


def run_report(inspection_output):
    """Run the Explanation Agent to generate an AI report.

    This is called separately (on button click) so the user
    doesn't wait for LLM response during the main inspection.
    """
    return generate_report(
        inspection_output["results"],
        inspection_output["config"],
    )
