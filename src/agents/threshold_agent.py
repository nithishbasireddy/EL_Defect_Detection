import numpy as np
from .gemini_client import ask_gemini


def compute_distribution(values, name):
    """Compute statistical summary of measurements."""
    arr = np.array(values)

    return f"""
{name} Distribution:
  - Count: {len(arr)}
  - Min: {arr.min():.2f}, Max: {arr.max():.2f}
  - Mean: {arr.mean():.2f}, Std: {arr.std():.2f}
  - Median: {np.median(arr):.2f}
  - 25th pct: {np.percentile(arr, 25):.2f}
  - 75th pct: {np.percentile(arr, 75):.2f}
  - 90th pct: {np.percentile(arr, 90):.2f}
  - Zero count: {np.sum(arr == 0)} / {len(arr)}
"""


def recommend_thresholds(results):
    """Analyze defect distributions and recommend pass/fail thresholds."""
    if len(results) == 0:
        return "No cells to analyze."

    crack_lengths = [r["crack_length_mm"] for r in results]
    dark_percents = [r["dark_percent"] for r in results]

    crack_stats = compute_distribution(crack_lengths, "Crack Length (mm)")
    dark_stats = compute_distribution(dark_percents, "Dark Area (%)")

    prompt = f"""You are a quality control engineer for solar panel manufacturing.
Given the defect distributions below, recommend optimal pass/fail thresholds.

{crack_stats}
{dark_stats}

Recommend:
1. **Crack Length Threshold** (mm) — with reasoning
2. **Dark Area Threshold** (%) — with reasoning
3. Brief overall module quality assessment

Consider industry standards and the data distribution shape.
Be concise.
"""

    return ask_gemini(prompt, temperature=0.2)
