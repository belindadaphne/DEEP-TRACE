import streamlit as st
import torch
from PIL import Image, ImageOps
import numpy as np
import tempfile
import os
import hashlib
import io
import csv
from datetime import datetime
import imageio.v2 as imageio

# ==========================================
# DEEP-TRACE: Deepfake Detection System
# ==========================================

st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🔍",
    layout="wide",
)

MODEL_PATH = "deeptrace_model_scripted.pt"
FRAME_SAMPLE_INTERVAL = 15
MAX_VIDEO_FRAMES = 300

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #050505, #160607);
    color: white;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}
.brand {
    font-size: 42px;
    font-weight: 800;
    color: white;
}
.brand span, .accent {
    color: #E50914;
}
.hero {
    padding: 35px 0 20px 0;
}
.hero-title {
    font-size: 48px;
    font-weight: 800;
}
.hero-text {
    color: #bbbbbb;
    font-size: 17px;
    line-height: 1.7;
}
.metric {
    background: #181818;
    border: 1px solid #333333;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
}
.stButton > button {
    background: #E50914;
    color: white;
    border-radius: 5px;
    border: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand">DEEP<span>-</span>TRACE</div>
<div style="color:#aaa;letter-spacing:2px;">
AI-POWERED MEDIA AUTHENTICITY ANALYSIS
</div>

<div class="hero">
    <div class="hero-title">
        Is it <span class="accent">REAL</span> or
        <span class="accent">FAKE?</span>
    </div>
    <div class="hero-text">
        Analyze images and videos using deep learning.
        Review frame-level predictions and download forensic reports.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"

    try:
        model = torch.jit.load(MODEL_PATH, map_location="cpu")
        model.eval()
        return model, None
    except Exception as exc:
        return None, str(exc)


model, model_error = load_model()

if model is None:
    st.error("Detection model could not be loaded.")
    st.code(model_error or "Unknown error")
else:
    st.success("Detection model loaded.")


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB").resize((224, 224))

    array = np.asarray(image, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    array = (array - mean) / std

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return tensor.float()


# ==========================================
# PREDICTION
# Class 0 = FAKE
# Class 1 = REAL
# ==========================================

def predict_image(image):
    if model is None:
        raise RuntimeError("Model is not available.")

    tensor = preprocess_image(image)

    with torch.inference_mode():
        output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if output.ndim != 2 or output.shape[-1] != 2:
        raise RuntimeError(
            f"Expected output shape [batch, 2], got {tuple(output.shape)}"
        )

    probabilities = torch.softmax(output, dim=1)[0].numpy()

    fake_score = float(probabilities[0])
    real_score = float(probabilities[1])

    verdict = "FAKE" if fake_score > real_score else "REAL"
    confidence = max(fake_score, real_score)

    return verdict, confidence, fake_score, real_score


# ==========================================
# DISPLAY RESULTS
# ==========================================

def display_result(verdict, confidence, fake_score, real_score):
    if verdict == "FAKE":
        st.error(
            f"⚠️ CLASSIFIED AS FAKE — {confidence * 100:.2f}% top-class score"
        )
    else:
        st.success(
            f"✓ CLASSIFIED AS REAL — {confidence * 100:.2f}% top-class score"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="metric"><h2>{confidence*100:.1f}%</h2>'
            '<p>TOP CLASS SCORE</p></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div class="metric"><h2>{real_score*100:.1f}%</h2>'
            '<p>REAL SCORE</p></div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f'<div class="metric"><h2>{fake_score*100:.1f}%</h2>'
            '<p>FAKE SCORE</p></div>',
            unsafe_allow_html=True,
        )

    st.progress(fake_score, text=f"FAKE score: {fake_score*100:.1f}%")
    st.progress(real_score, text=f"REAL score: {real_score*100:.1f}%")


# ==========================================
# FORENSIC REPORT
# ==========================================

def calculate_hash(data):
    return hashlib.sha256(data).hexdigest()


def create_report(
    filename,
    media_type,
    verdict,
    confidence,
    fake_score,
    real_score,
    file_hash,
    details,
):
    return f"""
DEEP-TRACE — MEDIA ANALYSIS REPORT
=================================

Generated: {datetime.now().isoformat(timespec="seconds")}
File name: {filename}
Media type: {media_type}
SHA-256: {file_hash}

Classification: {verdict}
Top-class score: {confidence*100:.2f}%
REAL score: {real_score*100:.2f}%
FAKE score: {fake_score*100:.2f}%

Analysis details:
{details}

LIMITATIONS
-----------
This is an AI model prediction, not proof of authenticity or manipulation.
Scores may not be calibrated probabilities. Independent verification
is recommended before drawing conclusions.
"""


# ==========================================
# MEDIA TYPE SELECTION
# ==========================================

media_type = st.radio(
    "Choose media type",
    ["Image", "Video"],
    horizontal=True,
)

st.divider()


# ==========================================
# IMAGE DETECTION
# ==========================================

if media_type == "Image":

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is not None:
        raw_data = uploaded.getvalue()

        try:
            image = Image.open(io.BytesIO(raw_data))
            image = ImageOps.exif_transpose(image).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True,
            )

            st.caption(
                f"File: {uploaded.name} | "
                f"Size: {len(raw_data)/1024:.1f} KB"
            )

            if st.button("🔍 ANALYZE IMAGE", disabled=model is None):

                with st.spinner("Analyzing image..."):
                    verdict, confidence, fake_score, real_score = (
                        predict_image(image)
                    )

                display_result(
                    verdict,
                    confidence,
                    fake_score,
                    real_score,
                )

                report = create_report(
                    uploaded.name,
                    "Image",
                    verdict,
                    confidence,
                    fake_score,
                    real_score,
                    calculate_hash(raw_data),
                    "Single image classified at 224 x 224 input resolution.",
                )

                st.download_button(
                    "⬇️ Download forensic report",
                    report,
                    file_name="deeptrace_image_report.txt",
                    mime="text/plain",
                )

        except Exception as exc:
            st.error("Could not process the image.")
            st.code(str(exc))


# ==========================================
# VIDEO DETECTION
# ==========================================

else:

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
    )

    if uploaded is not None:

        raw_data = uploaded.getvalue()

        st.video(raw_data)

        st.caption(
            f"File: {uploaded.name} | "
            f"Size: {len(raw_data)/(1024*1024):.2f} MB"
        )

        st.info(
            f"Frame sampling interval: every {FRAME_SAMPLE_INTERVAL} frames. "
            f"Maximum decoded frames: {MAX_VIDEO_FRAMES}."
        )

        if st.button("🔍 ANALYZE VIDEO", disabled=model is None):

            video_path = None
            frame_results = []

            try:
                suffix = os.path.splitext(uploaded.name)[1] or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp:
                    temp.write(raw_data)
                    video_path = temp.name

                reader = imageio.get_reader(video_path)

                progress = st.progress(0)

                try:
                    for index, frame in enumerate(reader):

                        if index >= MAX_VIDEO_FRAMES:
                            break

                        if index % FRAME_SAMPLE_INTERVAL == 0:
                            try:
                                frame_image = Image.fromarray(frame).convert("RGB")

                                verdict, confidence, fake_score, real_score = (
                                    predict_image(frame_image)
                                )

                                frame_results.append({
                                    "Frame": index,
                                    "Timestamp (seconds)": None,
                                    "Prediction": verdict,
                                    "Top score (%)": round(confidence * 100, 2),
                                    "FAKE score (%)": round(fake_score * 100, 2),
                                    "REAL score (%)": round(real_score * 100, 2),
                                })

                            except Exception:
                                pass

                        if index % 10 == 0:
                            progress.progress(
                                min(1.0, (index + 1) / MAX_VIDEO_FRAMES)
                            )

                finally:
                    reader.close()

                progress.progress(1.0)

                if not frame_results:
                    st.error("No frames could be analyzed.")

                else:
                    fake_score = float(np.mean([
                        row["FAKE score (%)"] for row in frame_results
                    ])) / 100.0

                    real_score = float(np.mean([
                        row["REAL score (%)"] for row in frame_results
                    ])) / 100.0

                    verdict = (
                        "FAKE" if fake_score > real_score else "REAL"
                    )

                    confidence = max(fake_score, real_score)

                    display_result(
                        verdict,
                        confidence,
                        fake_score,
                        real_score,
                    )

                    st.subheader("Sampled frame results")

                    # Use a conservative FPS assumption only if the reader
                    # exposes valid metadata. Otherwise show frame index.
                    for row in frame_results:
                        row["Timestamp (seconds)"] = "Not available"

                    st.dataframe(
                        frame_results,
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Download CSV results
                    import pandas as pd

                    csv_data = pd.DataFrame(frame_results).to_csv(index=False)

                    st.download_button(
                        "⬇️ Download frame results (CSV)",
                        csv_data,
                        file_name="deeptrace_frame_results.csv",
                        mime="text/csv",
                    )

                    report = create_report(
                        uploaded.name,
                        "Video",
                        verdict,
                        confidence,
                        fake_score,
                        real_score,
                        calculate_hash(raw_data),
                        (
                            f"Analyzed {len(frame_results)} sampled frames. "
                            f"Sampling interval: {FRAME_SAMPLE_INTERVAL}."
                        ),
                    )

                    st.download_button(
                        "⬇️ Download forensic report",
                        report,
                        file_name="deeptrace_video_report.txt",
                        mime="text/plain",
                    )

            except Exception as exc:
                st.error("Video analysis failed.")
                st.code(str(exc))

            finally:
                if video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                    except OSError:
                        pass


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center;color:#888;padding:20px;">
    <h3>DEEP-TRACE</h3>
    AI-Powered Deepfake Detection Prototype<br>
    <small>
    For educational use. Predictions are not forensic proof.
    </small>
</div>
""", unsafe_allow_html=True)
