import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Drought Prediction Dashboard", layout="wide")
st.title("🌾 Maharashtra Drought Prediction Dashboard")

# -----------------------------
# Config
# -----------------------------
RESULTS_DIR = "."  # same folder as app.py

# Helper to load image
def load_image(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        st.error(f"❌ File not found: {filename}")
        return None
    return Image.open(path)

# -----------------------------
# Section 1: Model Performance
# -----------------------------
st.subheader("📊 Model Performance Comparison")
img_perf = load_image("model_performance_comparison_dual.png")
if img_perf:
    st.image(img_perf, use_column_width=True)

# -----------------------------
# Section 2: Confusion Matrices
# -----------------------------
st.subheader("📈 Confusion Matrices")
img_cm = load_image("confusion_matrices_ts.png")
if img_cm:
    st.image(img_cm, use_column_width=True)

# -----------------------------
# Section 3: Drought Class Distribution
# -----------------------------
st.subheader("🧮 Drought Class Distribution")
img_dist = load_image("drought_class_distribution_ts.png")
if img_dist:
    st.image(img_dist, use_column_width=True)

