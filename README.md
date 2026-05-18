# EL Defect Detection

This repository contains a pipeline for detecting and analyzing defects in solar cells using Electroluminescence (EL) imaging. It uses a U-Net model built with PyTorch for semantic segmentation and includes a Streamlit frontend for batch processing and QA reporting.

## Overview

The system processes EL images to find specific types of defects: micro-cracks, dark areas, and finger interruptions. After segmentation, it runs physical analysis (like measuring crack lengths in millimeters) and uses a decision agent to automatically flag panels as PASS or FAIL based on configurable thresholds.

## Features

- **Defect Segmentation:** U-Net architecture trained with a combined Dice-Focal loss using `segmentation-models-pytorch`.
- **Crack Analysis:** Uses skeletonization (`scikit-image`) to measure the exact length and severity of detected cracks.
- **Dark Area Detection:** Intensity-based detection that supports optional reference-based calibration to account for lighting differences.
- **Decision Engine:** Evaluates the defect metrics against limits (e.g., `max_crack_length`) and uses the Groq API to generate human-readable verdicts.
- **Streamlit UI:** A local web interface for uploading images, visualizing the predicted masks alongside the ground truth, and exporting batch results to CSV.

## Repository Structure

```text
EL_Defect_Detection/
├── checkpoints/              # Model weights (.pth)
├── data/                     # Raw and processed EL images
├── src/                      # Source code
│   ├── agents/               # Decision logic and LLM integrations
│   ├── datasets/             # PyTorch dataset classes
│   ├── losses/               # Custom loss functions
│   ├── models/               # Model definitions
│   ├── pipeline/             # Inference wrappers
│   └── utils/                # Image processing and stat scripts
├── .env                      # API keys (not tracked in git)
├── requirements.txt          # Python dependencies
└── ...                       # Various test and evaluation scripts
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nithishbasireddy/EL_Defect_Detection.git
   cd EL_Defect_Detection
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows:
   .venv\Scripts\activate
   
   # Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory and add your Groq API key (needed for the decision agent):
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

### Web Interface
To start the Streamlit application for image processing and visualization:
```bash
streamlit run app.py
```
This will open the dashboard in your browser (default: `http://localhost:8501`).

### Command Line Scripts
You can also run individual components of the pipeline directly:

```bash
# Test the crack analysis functions
python test_crack_analysis.py

# Run prediction visualizations
python src/visualize_predictions.py
```

## Output & Reports
When running batch processing through the UI, the system outputs the percentage of dark areas, the longest crack dimension, and the detected classes. This data, along with the final PASS/FAIL verdict, can be exported as a CSV file for auditing.
