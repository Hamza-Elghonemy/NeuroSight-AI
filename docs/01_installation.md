# Installation Guide

This guide provides detailed instructions on how to set up the **NeuroSight-AI** environment on your local machine.

## Prerequisites

Before proceeding, ensure you have the following software installed:

- **Python 3.8 or higher**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **pip** (Python package installer): Usually comes with Python.

## Step 1: Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/Hamza-Elghonemy/NeuroSight-AI.git
cd FinalProject
```

## Step 2: Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other projects.

### On macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### On Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal line, indicating the environment is active.

## Step 3: Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### Note on PyTorch & Hardware Acceleration

The default installation command installs the standard PyTorch version. For optimal performance, you may want to install a version specific to your hardware:

- **Mac (Apple Silicon M1/M2/M3)**: Newer versions of PyTorch support MPS (Metal Performance Shaders).
- **NVIDIA GPU (Windows/Linux)**: Ensure you have `CUDA` drivers installed for GPU acceleration.
- **CPU Only**: No special action needed, but inference will be slower.

To verify your PyTorch device support, you can run this Python snippet:

```python
import torch
if torch.cuda.is_available():
    print("CUDA is available (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    print("MPS is available (Apple Silicon)")
else:
    print("Using CPU")
```

## Step 4: Verify Installation

Ensure all files are in place:

```bash
ls -R
```

You should see folders like `src`, `data`, `models`, `notebooks`, and files like `app.py`.

## Troubleshooting

- **"Module not found" error**: Ensure you activated your `venv` before running the app.
- **Slow performance**: Check if your device is using CPU instead of GPU/MPS.
- **Installation fails**: Try upgrading pip: `pip install --upgrade pip` and run the install command again.
