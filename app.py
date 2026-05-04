import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("Brain Tumor Classification")

# -------------------------
# CLASS NAMES
# -------------------------
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load("resnet502A.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------------------------
# UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert('RGB')

    input_tensor = transform(image).unsqueeze(0)

    # -------------------------
    # PREDICTION
    # -------------------------
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100

    # -------------------------
    # RESULT
    # -------------------------
    st.header("Prediction")
    st.markdown(f"## Class: {class_names[pred]}")
    st.markdown(f"### Confidence: {confidence:.2f}%")

    # -------------------------
    # PROBABILITIES (FIXED ORDER)
    # -------------------------
    st.subheader("Class Probabilities")

    for i, cls in enumerate(class_names):
        percentage = probs[0][i].item() * 100
        
        # name first
        st.write(f"{cls}: {percentage:.2f}%")
        
        # then bar
        st.progress(float(probs[0][i]))

    st.markdown("---")

    # -------------------------
    # GRAD-CAM
    # -------------------------
    rgb_img = np.array(image.resize((224, 224))) / 255.0
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() + 1e-8)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width=300)

    with col2:
        st.subheader("Grad-CAM")
        st.image(cam_image, width=300)