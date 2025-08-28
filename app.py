# app.py
# Streamlit Rapid Prototype: Image Processing with Webcam/URL/Upload
# Features:
# - Capture from webcam (st.camera_input) or load from URL or upload file
# - Simple image processing: grayscale, Gaussian blur, Canny edges
# - GUI controls for parameters
# - Display processed image
# - Plot histogram of pixel intensities

import io
import requests
import numpy as np
import streamlit as st
import cv2
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Rapid Prototype - Image Processing", layout="wide")
st.title("AI Rapid Prototype: Image Processing Demo (Streamlit)")

with st.sidebar:
    st.header("📥 Input Source")
    source = st.radio("Choose image source:", ["Webcam", "Upload", "Image URL"], index=0)

    st.header("⚙️ Processing Options")
    proc = st.selectbox("Processing type", ["None", "Grayscale", "Gaussian Blur", "Canny Edge"], index=1)

    # Common controls
    resize_width = st.slider("Resize width (px)", min_value=256, max_value=1920, value=640, step=32)

    # Gaussian blur params
    blur_ksize = st.slider("Gaussian blur kernel (odd)", min_value=1, max_value=51, value=5, step=2)
    blur_sigma = st.slider("Gaussian sigma", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # Canny params
    canny_low = st.slider("Canny threshold1", 0, 255, 100)
    canny_high = st.slider("Canny threshold2", 0, 255, 200)

    st.markdown("---")
    st.caption("Tip: Try toggling processing type and parameters to see changes live.")

def load_image_from_url(url: str):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")
        return None

def pil_to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv: np.ndarray):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def ensure_width(img: Image.Image, width: int):
    if width is None or width <= 0:
        return img
    w, h = img.size
    if w == width:
        return img
    ratio = width / float(w)
    new_size = (width, int(h * ratio))
    return img.resize(new_size, Image.LANCZOS)

col_left, col_right = st.columns([1,1])

# --- Acquire image ---
img_pil = None
if source == "Webcam":
    cam = st.camera_input("Capture image from webcam")
    if cam is not None:
        img_pil = Image.open(cam).convert("RGB")
elif source == "Upload":
    uploaded = st.file_uploader("Upload an image (png/jpg/jpeg)", type=["png","jpg","jpeg"])
    if uploaded is not None:
        img_pil = Image.open(uploaded).convert("RGB")
else:  # Image URL
    url = st.text_input("Enter image URL", value="https://picsum.photos/800")
    if st.button("Load from URL"):
        img_pil = load_image_from_url(url)

if img_pil is None:
    st.info("Provide an image from webcam, upload, or URL to begin.")
    st.stop()

# --- Resize ---
img_pil = ensure_width(img_pil, resize_width)

# --- Processing ---
img_cv = pil_to_cv(img_pil)
processed_cv = img_cv.copy()

if proc == "Grayscale":
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    processed_cv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
elif proc == "Gaussian Blur":
    k = max(1, blur_ksize // 2 * 2 + 1)  # ensure odd
    processed_cv = cv2.GaussianBlur(img_cv, (k, k), blur_sigma)
elif proc == "Canny Edge":
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=canny_low, threshold2=canny_high)
    processed_cv = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

processed_pil = cv_to_pil(processed_cv)

# --- Display images ---
with col_left:
    st.subheader("Original")
    st.image(img_pil, use_column_width=True)
with col_right:
    st.subheader("Processed")
    st.image(processed_pil, use_column_width=True)

# --- Plot a graph from image properties: intensity histogram ---
st.subheader("📊 Intensity Histogram")
fig = plt.figure()
gray_for_hist = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2GRAY)
plt.hist(gray_for_hist.ravel(), bins=32, range=(0,255))
plt.xlabel("Intensity (0-255)")
plt.ylabel("Frequency")
plt.title("Histogram of Processed Image")
st.pyplot(fig, use_container_width=True)

# --- Extra diagnostics ---
st.markdown("**Image Stats (Processed, Grayscale)**")
st.json({
    "width": processed_pil.width,
    "height": processed_pil.height,
    "mean_intensity": float(np.mean(gray_for_hist)),
    "std_intensity": float(np.std(gray_for_hist)),
})
