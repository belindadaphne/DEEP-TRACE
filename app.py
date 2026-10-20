import os
import io
import json
import hashlib
import tempfile
from datetime import datetime, timezone

import streamlit as st
import torch
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="DEEP-TRACE", page_icon="🔎", layout="wide")

st.title("🔎 DEEP-TRACE")
st.caption("AI-assisted deepfake screening")

MODEL_PATH = "deeptrace_model_scripted.pt"
FAKE_INDEX = 0
REAL_INDEX = 1
UNCERTAIN_MARGIN = 0.10


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model


try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image):
    tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise ValueError("Unsupported model output")

    output = output.detach().float().cpu()

    if output.ndim == 2:
        output = output[0]

    raw = output.numpy().astype(float)

    if len(raw) != 2:
        raise ValueError(f"Expected 2 outputs, got {len(raw)}")

    if (
        np.all(raw >= 0)
        and np.all(raw <= 1)
        and np.isclose(raw.sum(), 1.0, atol=1e-3)
    ):
        scores = raw
    else:
        ex = np.exp(raw - np.max(raw))
        scores = ex / ex.sum()

    fake = float(scores[FAKE_INDEX])
    real = float(scores[REAL_INDEX])

    if abs(fake - real) < UNCERTAIN_MARGIN:
        verdict = "UNCERTAIN"
    else:
        verdict = "FAKE" if fake > real else "REAL"

    return {
        "verdict": verdict,
        "fake_score": fake,
        "real_score": real,
        "raw_output": raw.tolist()
    }


def file_hash(data):
    return hashlib.sha256(data).hexdigest()


def make_report(data, filename, button_label):
    st.download_button(
        button_label,
        data=json.dumps(data, indent=2),
        file_name=filename,
        mime="application/json"
    )


# ---------------- IMAGE DETECTION ----------------

st.header("1. Image Detection")

image_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "webp"],
    key="image"
)

if image_file:
    image_bytes = image_file.getvalue()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(image, width=350)

        if st.button("Analyze Image", type="primary"):
            result = predict_image(image)

            st.subheader(f"Result: {result['verdict']}")

            c1, c2 = st.columns(2)
            c1.metric("REAL", f"{result['real_score'] * 100:.2f}%")
            c2.metric("FAKE", f"{result['fake_score'] * 100:.2f}%")

            with st.expander("Diagnostic model output"):
                st.write(result["raw_output"])
                st.caption(
                    "Class order is assumed to be FAKE then REAL. "
                    "Verify against the training code."
                )

            make_report(
                {
                    "file": image_file.name,
                    "type": "image",
                    "sha256": file_hash(image_bytes),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    **result
                },
                "deep_trace_image_report.json",
                "Download image report"
            )

    except Exception as e:
        st.error(f"Image analysis failed: {e}")


# ---------------- VIDEO DETECTION ----------------

st.divider()
st.header("2. Video Detection")

video_file = st.file_uploader(
    "Upload video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    key="video"
)

if video_file:
    video_bytes = video_file.getvalue()
    st.video(video_bytes)

    if st.button("Analyze Video", type="primary"):
        temp_path = None
        reader = None

        try:
            suffix = os.path.splitext(video_file.name)[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as tmp:
                tmp.write(video_bytes)
                temp_path = tmp.name

            reader = imageio.get_reader(temp_path, format="ffmpeg")
            metadata = reader.get_meta_data()
            fps = float(metadata.get("fps") or 0)
            every = max(1, round
