import os
import json
import gdown

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# GOOGLE DRIVE DOWNLOAD SYSTEM
# ------------------------------------------------------------------------------
MODEL_PATH = "models/tin_shed_resnet18.pth"
CLASS_IDX_PATH = "models/class_indices.json"

# Insert your real Google Drive file ID here:
FILE_ID = "1FyPgn1nTa73BMXij9nT__AsstHzRkeXe"

# Direct download URL:
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Ensure models/ exists
os.makedirs("models", exist_ok=True)

# Download model automatically if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Model downloaded.")

# ------------------------------------------------------------------------------


@st.cache_resource
def load_model_and_classes():
    # Load class labels
    with open(CLASS_IDX_PATH, "r") as f:
        raw = json.load(f)
    idx_to_class = {int(k): v for k, v in raw.items()}
    num_classes = len(idx_to_class)

    # Load ResNet18
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except:
        model = models.resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, idx_to_class


def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def main():
    st.set_page_config(page_title="Tin Shed Corrosion Detection")
    st.title("🧪 Tin Shed Corrosion Detection")
    st.write("Upload an image to classify corrosion level.")

    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        model, idx_to_class = load_model_and_classes()
        x = preprocess(img)

        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)[0]
            pred = torch.argmax(probs).item()

        pred_class = idx_to_class[pred]

        st.subheader("Prediction")
        st.write(f"**Corrosion Level: {pred_class}**")

        st.subheader("Probabilities")
        for i in range(len(idx_to_class)):
            st.write(f"{idx_to_class[i]}: {probs[i].item():.4f}")

        st.subheader("Recommendation")
        if pred_class == "non_damaged":
            st.success("Tin shed is in **good condition**.")
        elif pred_class == "semi_damaged":
            st.warning("Tin shed is **moderately corroded**. Maintenance recommended.")
        else:
            st.error("Tin shed has **severe corrosion**. Immediate repair recommended.")


if __name__ == "__main__":
    main()
