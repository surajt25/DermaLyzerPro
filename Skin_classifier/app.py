import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Skin Condition Classifier", layout="centered")


# Model definitions (match training)
class SkinCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class EfficientNetWrapper(nn.Module):
    def __init__(self, num_classes=7):
        super(EfficientNetWrapper, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.model(x)


# Configurations

device = "cuda" if torch.cuda.is_available() else "cpu"

# Make sure this order matches your ImageFolder class order used during training:
class_names = ["acne", "eczema", "lichen", "psoriasis", "rosacea", "vitiligo", "warts"]
num_classes = len(class_names)

# Paths to saved weights (prefer _best.pth if available)
CNN_WEIGHTS = r"checkpoints\custom_model_best.pth"
EFF_WEIGHTS = r"checkpoints\efficientnet_model_best.pth"


# Loading models

@st.cache_resource(show_spinner=False)
def load_models():
    models_dict = {}

    cnn = SkinCNN(num_classes=num_classes).to(device)
    if os.path.exists(CNN_WEIGHTS):
        cnn.load_state_dict(torch.load(CNN_WEIGHTS, map_location=device))
    else:
        st.warning(f"Custom CNN weights not found at: {CNN_WEIGHTS}")
    cnn.eval()
    models_dict["cnn"] = cnn

    eff = EfficientNetWrapper(num_classes=num_classes).to(device)
    if os.path.exists(EFF_WEIGHTS):
        eff.load_state_dict(torch.load(EFF_WEIGHTS, map_location=device))
    else:
        st.warning(f"EfficientNet weights not found at: {EFF_WEIGHTS}")
    eff.eval()
    models_dict["eff"] = eff

    return models_dict

models_loaded = load_models()


# Transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Inference helpers

def predict_single(model, image_tensor):
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

def predict(image: Image.Image, use_cnn: bool, use_eff: bool):
    x = transform(image).unsqueeze(0).to(device)

    probs_list = []
    if use_cnn:
        probs_list.append(predict_single(models_loaded["cnn"], x))
    if use_eff:
        probs_list.append(predict_single(models_loaded["eff"], x))

    if len(probs_list) == 0:
        return None, None, None

    if len(probs_list) == 1:
        probs = probs_list[0]
    else:
        probs = np.mean(probs_list, axis=0)  # simple average ensemble

    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], float(probs[pred_idx]), probs

def plot_probs(probs):
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, probs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlim([0, 1])
    ax.set_xlabel("Probability")
    ax.set_title("Class Probabilities")
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v*100:.1f}%", va='center')
    fig.tight_layout()
    return fig


# User Interface

st.title(" Skin Condition Classifier")
st.caption(f"Device: **{device}** | Models: Custom CNN + EfficientNet-B0 | Ensemble averaging")

col1, col2 = st.columns(2)
use_cnn = col1.checkbox("Use Custom CNN", True)
use_eff = col2.checkbox("Use EfficientNet-B0", True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        pred, conf, probs = predict(image, use_cnn, use_eff)
        if probs is None:
            st.error("Please enable at least one model.")
        else:
            st.success(f"Prediction: **{pred}** ({conf*100:.2f}% confidence)")
            st.pyplot(plot_probs(probs))

            # Optional: show top-3
            top3_idx = np.argsort(probs)[::-1][:3]
            st.write("**Top-3 classes:**")
            for i in top3_idx:
                st.write(f"- {class_names[i]}: {probs[i]*100:.2f}%")
