import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import pandas as pd
import glob

from src.pipeline.full_pipeline import process_image
from src.pipeline.inference import load_model, predict_cell
from src.utils.visualization import overlay_mask
from src.agents.orchestrator import run_inspection, run_report
from src.agents.threshold_agent import recommend_thresholds


# -----------------------
# PAGE
# -----------------------
st.title("EL Defect Detection System")

# -----------------------
# USER CONFIG
# -----------------------
st.sidebar.header("Inspection Settings")

# Crack length
col1, col2 = st.sidebar.columns(2)
max_crack_length_slider = col1.slider("Crack Length (mm)", 1, 50, 20)
max_crack_length = col2.number_input("Manual", 1, 50, max_crack_length_slider)

# Crack count
col1, col2 = st.sidebar.columns(2)
max_crack_count_slider = col1.slider("Max Cracks", 1, 10, 3)
max_crack_count = col2.number_input("Manual ", 1, 10, max_crack_count_slider)

# Dark area
col1, col2 = st.sidebar.columns(2)
max_dark_slider = col1.slider("Dark Area (%)", 1, 30, 10)
max_dark_percent = col2.number_input("Manual  ", 1, 30, max_dark_slider)

detect_finger = st.sidebar.checkbox("Detect Finger Interruption")
detect_foreign = st.sidebar.checkbox("Detect Foreign Objects")

# -----------------------
# AI SMART THRESHOLDS
# -----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("AI Agent")

if st.sidebar.button("Suggest Smart Thresholds"):
    if "inspection_output" in st.session_state:
        with st.sidebar:
            with st.spinner("AI analyzing defect distribution..."):
                suggestion = recommend_thresholds(
                    st.session_state["inspection_output"]["results"]
                )
            st.markdown(suggestion)
    else:
        st.sidebar.warning("Run an inspection first!")


# -----------------------
# LOAD MODEL
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    classes=5
)

model = load_model(model, "checkpoints/best_model.pth", device)

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])


# -----------------------
tab_single, tab_batch = st.tabs(['Single Image Inspection', 'Batch Processing'])

with tab_single:
    # UPLOAD IMAGE
    # -----------------------
    uploaded_file = st.file_uploader("Upload EL Image", type=["jpg", "png"])
    ref_file = st.file_uploader("Upload Reference (Good Module)", type=["jpg", "png"])

    if uploaded_file:
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.read())

        st.image("temp.jpg", caption="Uploaded Image", width="stretch")

        show_overlay = st.checkbox("Show Defect Overlay")

        if st.button("Run Inspection"):

            # Handle reference image
            ref_mean = None

            if ref_file is not None:
                with open("ref.jpg", "wb") as f:
                    f.write(ref_file.read())

                ref_img = cv2.imread("ref.jpg")
                ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                ref_mean = float(np.mean(ref_gray))

            # --- Run Pipeline ---
            pipeline_results = process_image("temp.jpg", model, transform, device)

            # --- Run Orchestrator ---
            config = {
                "max_crack_length": max_crack_length,
                "max_dark_percent": max_dark_percent,
            }

            inspection = run_inspection(pipeline_results, config, ref_mean)

            # Save for UI rendering outside the button block
            st.session_state["inspection_output"] = inspection
            st.session_state["ref_mode"] = "reference" if ref_file is not None else "model"
            st.session_state["inspection_run"] = True

    # --- Display Results ---
    if st.session_state.get("inspection_run"):
        inspection = st.session_state["inspection_output"]
        results = inspection["results"]
        summary = inspection["summary"]
    
        if st.session_state.get("ref_mode") == "reference":
            st.info("Using reference-based dark detection")
        else:
            st.warning("Using model-based dark detection")

        st.markdown("---")

        # Full Module Overlay Button (Moved here per user request)
        if st.button("View Full Module Defect Overlay"):
            if summary['total_cells'] == 1:
                st.info("The uploaded image is already a single cell. The defect overlay is shown below!")
            else:
                with st.spinner("Generating full module overlay..."):
                    full_img = cv2.imread("temp.jpg")
                    full_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
                
                    raw_mask = predict_cell(model, full_rgb, transform, device)
                    from src.pipeline.mask_cleaning import clean_mask_classwise
                    clean_mask = clean_mask_classwise(raw_mask)
                
                    full_overlay = overlay_mask(full_img, clean_mask)
                    st.image(full_overlay, caption="Full Module Defect Overlay", use_container_width=True)

        st.markdown("---")

        # --- Display Cell Results ---
        st.subheader(f"Total Cells: {summary['total_cells']}")

        for r in results:
            st.write(
                f"Cell {r['cell_id']} → "
                f"Crack: {r['crack_length_mm']:.2f} mm | "
                f"Dark: {r['dark_percent']:.1f}% → "
                f"{r['verdict']}"
            )

            # --- Clean, collapsible button for individual lengths ---
            if "individual_cracks_mm" in r and len(r["individual_cracks_mm"]) > 0:
                with st.expander(f"View individual crack lengths for Cell {r['cell_id']}"):
                    for i, length in enumerate(r["individual_cracks_mm"], 1):
                        st.write(f"• Crack {i}: **{length} mm**")

            if show_overlay:
                overlay_img = overlay_mask(r["cell"], r["mask"])
                st.image(overlay_img, caption=f"Cell {r['cell_id']} Overlay", width="stretch")

        # --- Module Summary ---
        st.subheader("Module Summary")
        st.write(f"Defective Cells: {summary['defective_cells']} / {summary['total_cells']}")
        st.write(f"Total Crack Length: {summary['total_crack_length']:.2f} mm")

        if summary["module_verdict"] == "FAIL":
            st.error("FINAL RESULT: FAIL")
        else:
            st.success("FINAL RESULT: PASS")

    # -----------------------
    # AI REPORT
    # -----------------------
    if "inspection_output" in st.session_state:

        st.markdown("---")
        st.subheader("AI Inspection Report")

        if st.button("Generate AI Report"):
            with st.spinner("AI agent analyzing inspection..."):
                report = run_report(st.session_state["inspection_output"])
            st.markdown(report)
with tab_batch:
    st.subheader("Batch Processing")
    st.write("Upload multiple EL images and export the results to CSV.")
    
    uploaded_batch_files = st.file_uploader("Select Multiple Images", type=["jpg", "png"], accept_multiple_files=True)
    batch_ref_file = st.file_uploader("Upload Reference (Good Module) - Optional", type=["jpg", "png"], key="batch_ref")
    
    if st.button("Run Batch Inspection"):
        if not uploaded_batch_files:
            st.error("Please upload at least one image to begin batch processing.")
        else:
            st.info(f"Found {len(uploaded_batch_files)} images. Starting processing...")
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Handle reference image for batch
            batch_ref_mean = None
            if batch_ref_file is not None:
                with open("batch_ref.jpg", "wb") as f:
                    f.write(batch_ref_file.read())
                import cv2, numpy as np
                batch_ref_img = cv2.imread("batch_ref.jpg")
                batch_ref_gray = cv2.cvtColor(batch_ref_img, cv2.COLOR_BGR2GRAY)
                batch_ref_mean = float(np.mean(batch_ref_gray))
            
            batch_results = []
            
            config = {
                "max_crack_length": max_crack_length,
                "max_dark_percent": max_dark_percent,
            }
            
            for i, uploaded_file in enumerate(uploaded_batch_files):
                file_name = uploaded_file.name
                status_text.text(f"Processing {i+1}/{len(uploaded_batch_files)}: {file_name}")
                
                try:
                    # Save to temp file since process_image expects a path
                    with open("batch_temp.jpg", "wb") as f:
                        f.write(uploaded_file.read())
                    
                    # Process image
                    pipeline_results = process_image("batch_temp.jpg", model, transform, device)
                    inspection = run_inspection(pipeline_results, config, ref_mean=batch_ref_mean)  
                    
                    summary = inspection["summary"]
                    batch_results.append({
                        "Filename": file_name,
                        "Total Cells": summary["total_cells"],
                        "Defective Cells": summary["defective_cells"],
                        "Total Crack (mm)": round(summary["total_crack_length"], 2),
                        "Verdict": summary["module_verdict"]
                    })
                except Exception as e:
                    batch_results.append({
                        "Filename": file_name,
                        "Total Cells": "ERROR",
                        "Defective Cells": "ERROR",
                        "Total Crack (mm)": "ERROR",
                        "Verdict": f"ERROR: {str(e)}"
                    })
                
                progress_bar.progress((i + 1) / len(uploaded_batch_files))
            
            status_text.text("Batch processing complete!")
            
            # Convert to dataframe
            import pandas as pd
            df = pd.DataFrame(batch_results)
            st.dataframe(df)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='batch_inspection_results.csv',
                mime='text/csv',
            )