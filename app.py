import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time

# -----------------------------
# CONFIG & PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="Smart Crop Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
/* Set global font and background */
html, body, [class*="css"]  {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background-color: #050816;
    color: #E2E8F0;
}

/* Main title */
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #22c55e, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.sub-title {
    font-size: 1rem;
    color: #94A3B8;
}

/* Prediction card */
.pred-card {
    background: linear-gradient(135deg, #020617, #0f172a);
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    border: 1px solid #1f2937;
    box-shadow: 0 18px 35px rgba(15,23,42,0.9);
}

/* Metric badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: rgba(34,197,94,0.08);
    color: #4ade80;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
}

/* Confidence bar container */
.conf-bar {
    width: 100%;
    height: 10px;
    border-radius: 999px;
    background: #1f2937;
    margin-top: 0.4rem;
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #22d3ee);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid #111827;
}

/* File uploader text color fix */
span[data-baseweb="tag"] {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
    border-radius: 999px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# MODEL & TRANSFORMS
# -----------------------------
MODEL_PATH = "resnet50_plant_disease_best.pth"
CLASSES_PATH = "classes.txt"
IMAGE_SIZE = 224

@st.cache_resource
def load_class_names():
    with open(CLASSES_PATH, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

CLASS_NAMES = load_class_names()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, len(CLASS_NAMES))
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

def predict(image: Image.Image):
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].numpy()

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    # top-3 classes
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]
    return CLASS_NAMES[top_idx], top_prob, top3

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("### 🌿 Smart Crop Doctor")
    st.markdown(
        "Upload a leaf image and get an AI‑powered diagnosis of the disease "
        "with confidence scores."
    )
    st.markdown("---")
    st.markdown("**Model:** ResNet‑50 (Transfer Learning)")
    st.markdown("**Accuracy:** ~99.5% on validation set")
    st.markdown("**Dataset:** New Plant Diseases Dataset (Augmented)")
    st.markdown("---")
    st.markdown("Tips:")
    st.markdown("- Use clear, focused leaf images\n- Avoid multiple leaves in one image\n- Try both healthy and diseased leaves")

# -----------------------------
# MAIN LAYOUT
# -----------------------------
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown('<div class="main-title">Smart Crop Doctor</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Deep‑learning based crop leaf disease detection. '
        'Upload an image, and the model will instantly identify the disease.</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a leaf image (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        analyze = st.button("🔍 Analyze Leaf", use_container_width=True)

        if analyze:
            with st.spinner("Running diagnosis..."):
                start = time.time()
                label, conf, top3 = predict(image)
                elapsed = time.time() - start

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="pred-card">', unsafe_allow_html=True)
            st.markdown('<span class="badge">Prediction</span>', unsafe_allow_html=True)
            st.markdown(f"<h3 style='margin-top:0.6rem; color:#e5e7eb;'>{label}</h3>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='margin-bottom:0.2rem; color:#9ca3af;'>"
                f"Confidence: <b>{conf*100:.2f}%</b> &nbsp; | &nbsp; "
                f"Inference time: <b>{elapsed*1000:.1f} ms</b></p>",
                unsafe_allow_html=True,
            )

            # Confidence bar
            st.markdown('<div class="conf-bar"><div class="conf-fill" style="width: {}%;"></div></div>'.format(conf*100),
                        unsafe_allow_html=True)

            # Top‑3 table
            st.markdown("<br><b>Top‑3 predictions</b>", unsafe_allow_html=True)
            for cls, p in top3:
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; font-size:0.9rem; color:#cbd5f5;'>"
                    f"<span>{cls}</span><span>{p*100:.2f}%</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("⬆️ Upload a leaf image on the left to start the diagnosis.")

with col_right:
    st.markdown("### 📊 Model Insights")
    st.markdown(
        "- Trained on 38 different disease and healthy classes.\n"
        "- Uses transfer learning on ResNet‑50 for high accuracy.\n"
        "- Strong data augmentation improves generalization."
    )
    st.markdown("### 🔎 What this app does")
    st.markdown(
        "This interface is built to demonstrate : "
        "crop leaf disease prediction from images with a clear, modern UI."
    )
