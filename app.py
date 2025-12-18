import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from src.inference import load_model, get_transforms, GradCAM, CLASSES, DEVICE

st.set_page_config(page_title="Brain CT/MRI Classifier", layout="wide")

st.title("🧠 Brain CT/MRI Diagnostic System")
st.markdown("""
This system classifies brain scans into **Hemorrhagic Stroke**, **Ischemic Stroke**, or **Tumor**.
It provides **Grad-CAM** visualizations for explainability.
""")


st.sidebar.header("Settings")
model_type = st.sidebar.selectbox("Select Model", ["Standard CNN (ResNet)", "Concept Bottleneck Model (CBM)"])
enable_gradcam = st.sidebar.checkbox("Show Grad-CAM Explainability", value=True)
enable_segmentation = st.sidebar.checkbox("Show Lesion Segmentation (Bonus)", value=False)


@st.cache_resource
def get_loaded_model(m_type):
    if m_type == "Concept Bottleneck Model (CBM)":
        return load_model('cbm', 'models/best_cbm.pth')
    else:
        return load_model('cnn', 'models/best_cnn.pth')

model = get_loaded_model(model_type)


uploaded_file = st.file_uploader("Upload CT or MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        

    transform = get_transforms()
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    if st.button("Analyze Scan"):
        with st.spinner("Analyzing..."):
            if model_type == "Concept Bottleneck Model (CBM)":
                logits, concepts = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                
                st.sidebar.subheader("Latent Concepts")
                concept_vals = concepts[0].detach().cpu().numpy()
                st.sidebar.bar_chart(concept_vals)
            else:
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
            
            probs_np = probs.detach().cpu().numpy()[0]
            pred_idx = np.argmax(probs_np)
            pred_class = CLASSES[pred_idx]
            confidence = probs_np[pred_idx]
            
            st.success(f"**Prediction: {pred_class.upper()}** ({confidence*100:.2f}%)")
            
            st.write("### Confidence Scores")
            for i, cls in enumerate(CLASSES):
                st.progress(float(probs_np[i]), text=f"{cls}: {probs_np[i]*100:.1f}%")


            if enable_gradcam:
                with col2:
                    st.subheader("Explainability (Grad-CAM)")
                    target_layer = model.backbone.layer4 if hasattr(model, 'backbone') else model.layer4
                    gcam = GradCAM(model, target_layer)
                    heatmap, _ = gcam(img_tensor)
                    
                    img_np = np.array(image.resize((224, 224))) / 255.0
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    heatmap_float = heatmap.astype(np.float32) / 255.0
                    
                    overlay = heatmap_float * 0.4 + img_np * 0.6
                    overlay = np.clip(overlay, 0, 1)
                    
                    st.image(overlay, caption="Model Attention Map", clamp=True, use_container_width=True)
                    st.info("Red areas indicate regions contributing most to the prediction.")


            if enable_segmentation:
                with col2:
                    st.subheader("Lesion Segmentation")
                    unet_model = load_model('unet', 'models/best_unet.pth')
                    
                    with torch.no_grad():
                        seg_out = unet_model(img_tensor)
                        seg_prob = torch.sigmoid(seg_out).squeeze().cpu().numpy()
                    
                    mask = (seg_prob > 0.5).astype(np.float32)
                    
                    img_np = np.array(image.resize((224, 224))) / 255.0
                    green_mask = np.zeros_like(img_np)
                    green_mask[:, :, 1] = mask # Green channel
                    
                    seg_overlay = (img_np * 0.7 + green_mask * 0.3)
                    seg_final = np.where(mask[..., None] > 0, seg_overlay, img_np)
                    
                    st.image(seg_final, caption="Predicted Lesion Mask", clamp=True, use_container_width=True)

