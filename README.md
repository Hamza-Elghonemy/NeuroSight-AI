# Brain CT/MRI Diagnostic System

A comprehensive AI-powered diagnostic tool for classifying and analyzing brain CT/MRI scans. This system detects **Hemorrhagic Stroke**, **Ischemic Stroke**, and **Tumors**, providing clinician-friendly visualizations and pixel-level lesion segmentation.

## 🌟 Features

*   **Multi-Class Classification**: Accurately classifies scans into three critical categories:
    *   Hemorrhagic Stroke
    *   Ischemic Stroke
    *   Brain Tumor
*   **Explainable AI (Grad-CAM)**: Visualizes the model's focus regions to build trust and verify predictions.
*   **Lesion Segmentation (Bonus)**: Uses a U-Net architecture to precisely segment lesions (e.g., hemorrhages) from healthy tissue.
*   **Concept Bottleneck Model (CBM)**: Offers an interpretable alternative to standard Black-Box CNNs by learning intermediate concepts.
*   **Interactive Web App**: A user-friendly Streamlit interface for real-time analysis.

## 🛠️ Technology Stack

*   **Deep Learning**: PyTorch, torchvision, ResNet50, U-Net
*   **Web Framework**: Streamlit
*   **Image Processing**: OpenCV, PIL
*   **Visualization**: Matplotlib, Grad-CAM
*   **Device Acceleration**: MPS (Mac Silicon), CUDA (NVIDIA), CPU support

## 📂 Project Structure

```
├── app.py                  # Main Streamlit Application
├── requirements.txt        # Python Dependencies
├── data/                   # Dataset Directory
├── models/                 # Trained Model Weights (.pth)
├── notebooks/              # Jupyter Notebooks for Training & Experiments
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Model_Training_CNN.ipynb
│   ├── 03_Concept_Bottleneck_Model.ipynb
│   ├── 04_Explainability_GradCAM.ipynb
│   └── 05_Lesion_Segmentation.ipynb
└── src/
    └── inference.py        # Core Inference Logic & Utilities
```

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the Application
Launch the web interface:
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

### 4. Training Models (Optional)
If you wish to retrain the models, follow the notebooks in the `notebooks/` directory in sequential order.
*   **Note**: The segmentation module (`05_Lesion_Segmentation.ipynb`) uses a **Hybrid Data Strategy** to generate realistic training samples without needing massive datasets.

## 🧠 Model Details

### Standard CNN
*   **Architecture**: ResNet50 (Pretrained on ImageNet)
*   **Inputs**: 224x224 RGB Images
*   **Classes**: Hemorrhagic, Ischemic, Tumor

### Lesion Segmentation (U-Net)
*   **Architecture**: Classic U-Net with skip connections.
*   **Output**: Binary mask highlighting the lesion area.

## ⚠️ Note on Medical Advice
This tool is for **educational and research purposes only**. It should not be used as a primary diagnostic tool in a clinical setting without further validation and regulatory approval.
