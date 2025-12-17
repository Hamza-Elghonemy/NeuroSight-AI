# Development & Contribution Guide

This guide is for developers who wish to extend, retrain, or modify the NeuroSight-AI system.

## Project Structure

```
├── app.py                  # Streamlit entry point
├── requirements.txt        # Dependencies
├── data/                   # Place raw datasets here
├── models/                 # Store trained .pth files here
├── notebooks/              # Experiments and training loops
│   ├── 01_Data_Preprocessing.ipynb
│   ├── ...
├── src/
│   └── inference.py        # Shared model processing logic
└── docs/                   # Documentation
```

## Adding a New Model

To add a new architecture (e.g., EfficientNet):

1. **Update `src/inference.py`**:
   - Import the new model from `torchvision` or define the class.
   - Update `load_model` function to handle a new `model_type` string (e.g., `'efficientnet'`).
2. **Update `app.py`**:
   - Add the new option to the `st.sidebar.selectbox`.
   - Ensure the correct `.pth` path is passed to `load_model`.

## Retraining Models

Training logic is encapsulated in the Jupyter notebooks in `notebooks/`.

1. **Data Prep**: Run `01_Data_Preprocessing.ipynb` to organize your raw DICOM/JPG images into train/val folders.
2. **Training**: Run `02_Model_Training_CNN.ipynb`.
   - Adjust hyperparameters (Learning Rate, Batch Size) in the configuration cell.
   - The best model will be saved to `models/` automatically.
3. **Evaluation**: Use `04_Explainability_GradCAM.ipynb` to batch-test the new model and generate reports.

## Code Style

- **Python**: Follow PEP 8 guidelines.
- **Imports**: Group standard libraries first, then third-party (torch, numpy), then local (`src`).

## Testing

Currently, manual testing via the notebooks and app is used.
- **Sanity Check**: Run `app.py` and analyze a known sample image to verify the pipeline doesn't crash.
