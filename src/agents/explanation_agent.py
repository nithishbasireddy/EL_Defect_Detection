import numpy as np
from .gemini_client import ask_gemini


def build_context(results, config):
    """Build a structured text summary from inspection results."""
    total = len(results)
    failed = [r for r in results if r["verdict"] == "FAIL"]
    passed = [r for r in results if r["verdict"] == "PASS"]

    cracks = [r["crack_length_mm"] for r in results]
    darks = [r["dark_percent"] for r in results]

    context = f"""
=== EL MODULE INSPECTION DATA ===

Module Overview:
- Total cells: {total}
- Passed: {len(passed)}
- Failed: {len(failed)}
- Defect rate: {len(failed) / total * 100:.1f}%

Crack Statistics:
- Total crack length: {np.sum(cracks):.2f} mm
- Average per cell: {np.mean(cracks):.2f} mm
- Maximum: {np.max(cracks):.2f} mm

Dark Area Statistics:
- Average: {np.mean(darks):.1f}%
- Maximum: {np.max(darks):.1f}%

Thresholds:
- Crack limit: {config['max_crack_length']} mm
- Dark area limit: {config['max_dark_percent']}%
"""

    if failed:
        context += "\nFailed Cells:\n"
        for r in failed:
            context += f"  Cell {r['cell_id']}: {', '.join(r['reasons'])}\n"

    return context


def generate_report(results, config):
    """Generate a full inspection report using Gemini."""
    context = build_context(results, config)

    prompt = f"""You are a quality control engineer analyzing an EL 
(electroluminescence) image of a solar PV module. 

Write a professional inspection report with these sections:
1. **Summary** — overall pass/fail, key numbers
2. **Defect Analysis** — what defects were found, where, how severe
3. **Root Causes** — possible manufacturing/handling causes
4. **Recommendations** — actions to take

Keep it under 300 words. Use bullet points.

{context}
"""

    return ask_gemini(prompt, temperature=0.3)


def explain_cell(cell_data):
    """Generate a brief explanation for a single cell's result."""
    if cell_data["verdict"] == "PASS":
        return None

    prompt = f"""You are a solar panel quality engineer. 
A cell failed inspection with these issues:
{', '.join(cell_data['reasons'])}
Crack severity: {cell_data['severity']}

In 2 sentences, explain the likely cause and impact on module performance.
"""

    return ask_gemini(prompt, temperature=0.3)
