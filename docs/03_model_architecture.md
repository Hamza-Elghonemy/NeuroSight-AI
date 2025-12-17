# Model Architecture & Technical Details

NeuroSight-AI utilizes three distinct deep learning architectures to achieve robust diagnostics and explainability.

## 1. Classification Backbone: ResNet50

We use a **ResNet50** (Residual Network with 50 layers) as the primary backbone for feature extraction.

- **Pretraining**: The model is initialized with weights trained on ImageNet, allowing it to leverage general visual features (edges, textures) before fine-tuning on medical data.
- **Customization**: The final fully connected (FC) layer is replaced to output 3 neurons, corresponding to our classes:
    1. Hemorrhagic Stroke
    2. Ischemic Stroke
    3. Tumor
- **Input Resolution**: 224x224 RGB images.

## 2. Concept Bottleneck Model (CBM)

The CBM offers a step towards "Glass-Box" AI. Instead of mapping *Pixels -> Prediction*, it maps *Pixels -> Concepts -> Prediction*.

- **Structure**:
    - **Backbone**: ResNet50 modified to output `N` concepts instead of classes.
    - **Bottleneck**: A layer of neurons representing high-level concepts (e.g., "Mass effect", "Hyper-density").
    - **Classifier**: A final linear layer mapping concepts to the 3 diagnostic classes.
- **Benefits**: Users can inspect *why* a decision was made by looking at which concepts were activated.

## 3. Lesion Segmentation: U-Net

For precise localization, we employ a custom **U-Net** architecture.

- **Encoder (Downsampling)**: Captures context using convolutional blocks followed by max-pooling.
- **Decoder (Upsampling)**: Enables precise localization using transposed convolutions.
- **Skip Connections**: Concatenate high-resolution features from the encoder with the decoder to preserve spatial details lost during pooling.
- **Output**: A pixel-wise binary map (0 = healthy, 1 = lesion).

## 4. Explainability: Grad-CAM

**Gradient-weighted Class Activation Mapping (Grad-CAM)** allows us to visualize the model's focus.

- **Mechanism**:
    1. We compute the gradient of the predicted class score with respect to the feature maps of the last convolutional layer (`layer4` in ResNet).
    2. These gradients are pooled to obtain "importance weights" for each feature map.
    3. A weighted sum of feature maps is calculated and passed through a ReLU to obtain the coarse heatmap.
- **Result**: A heatmap where high intensity (Red) corresponds to regions that positively influenced the class prediction.

## 5. Data Flow Pipeline

1. **Preprocessing**:
   - Resize to 224x224.
   - Normalization (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).
   - Conversion to PyTorch Tensor.
2. **Inference**:
   - Image passed through CNN/CBM.
   - Softmax applied to logits to get probabilities.
3. **Post-processing**:
   - Grad-CAM heatmap generated (if enabled).
   - Segmentation mask generated (if enabled) and overlaid on the original image.
