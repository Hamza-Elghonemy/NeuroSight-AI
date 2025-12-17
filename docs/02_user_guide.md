# User Guide

This section explains how to use the **NeuroSight-AI** web application for diagnosing brain CT/MRI scans.

## Launching the Application

1. Activate your virtual environment (see [Installation Guide](01_installation.md)).
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. The application will open automatically in your default web browser at `http://localhost:8501`.

## Interface Overview

The interface consists of a **Sidebar** for settings and a **Main Area** for analysis.

### 1. Sidebar Settings

- **Select Model**: Choose the AI architecture for prediction.
  - **Standard CNN (ResNet)**: Best for general accuracy.
  - **Concept Bottleneck Model (CBM)**: Provides interpretable "concepts" (e.g., specific visual features) along with the prediction.
- **Show Grad-CAM Explainability**: Toggles the heatmap visualization that highlights which parts of the image influenced the decision.
- **Show Lesion Segmentation (Bonus)**: Toggles the pixel-level mask overlay for identifying specific lesion boundaries.

### 2. Uploading a Scan

- Click **"Browse files"** or drag and drop an image into the upload area.
- Supported formats: `.jpg`, `.png`, `.jpeg`.
- Once uploaded, the raw image will be displayed in the "Input Image" column.

### 3. Analyzing results

Click the **"Analyze Scan"** button to start the diagnostic process.

#### Prediction & Confidence
- The system displays the predicted class (**Hemorrhagic Stroke**, **Ischemic Stroke**, or **Tumor**).
- A confidence score (0-100%) indicates the model's certainty.
- A progress bar breakdown shows the probabilities for all three classes.

#### Latent Concepts (CBM Only)
- If you selected the **CBM** model, a chart in the sidebar will show the activation levels of internal "concepts". These represent intermediate features the model learned to recognize.

#### Explainability (Grad-CAM)
- **Visual**: A "heat map" overlay on your image.
- **Interpretation**: Red/Yellow areas are the "hotspots" the model focused on. Blue areas were ignored. If the model predicts a tumor, the red area should align with the tumor location. This helps verify that the AI isn't looking at artifacts (like text or skull borders).

#### Lesion Segmentation
- **Visual**: A green semi-transparent mask overlaid on the original image.
- **Interpretation**: This outlines the precise shape and location of the pathology (e.g., the bleed or the tumor mass). This is useful for sizing and localization.

## Example Workflow

1. **Upload** a scan of a suspected stroke patient.
2. Select **Standard CNN** for the initial check.
3. Enable **Grad-CAM**.
4. Click **Analyze**.
5. Result: "Hemorrhagic Stroke (98%)".
6. Check the **Grad-CAM** image: The red highlight is centered on a white hyperdense region in the brain tissue.
7. Checks out! The model is looking at the correct features.
