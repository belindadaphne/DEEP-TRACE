import os
import io
import json
import hashlib
import tempfile
from datetime import datetime, timezone

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 DEEP-TRACE")
st.caption("AI-powered image and video deepfake screening")


# ---------------- SETTINGS ----------------
MODEL_PATH = "deeptrace_model_scripted.pt"

# These labels preserve the current app's assumption.
# Confirm their order against the model's training code.
FAKE_INDEX = 0
REAL_INDEX = 1

# Small score differences produce an UNCERTAIN result.
UNCERTAIN_MARGIN = 0.10

DEVICE = torch.device("cpu")


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    loaded_model = torch.jit.load(
        MODEL_PATH,
        map_location=DEVICE
    )
    loaded_model.eval()
    return loaded_model


try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load detection model: {exc}")
    st.stop()


# ---------------- IMAGE PREPROCESSING ----------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------- PREDICTION FUNCTION ----------------
def predict_image(image):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise ValueError("Unsupported model output.")

    output = output.detach().float().cpu()

    if output.ndim == 1:
        output = output.unsqueeze(0)

    if output.ndim != 2 or output.shape[0] < 1:
        raise ValueError(
            f"Unexpected model output shape: {tuple(output.shape)}"
        )

    raw_values = output[0].numpy().astype(float)

    if len(raw_values) != 2:
        raise ValueError(
            f"Expected two class outputs, got {len(raw_values)}."
        )

    # Use probabilities directly if already normalized;
    # otherwise convert logits using softmax.
    if (
        np.all(raw_values >= 0)
        and np.all(raw_values <= 1)
        and np.isclose(raw_values.sum(), 1.0, atol=1e-3)
    ):
        scores = raw_values
    else:
        shifted = raw_values - np.max(raw_values)
        exponentials = np.exp(shifted)
        scores = exponentials / exponentials.sum()

    fake_score = float(scores[FAKE_INDEX])
    real_score = float(scores[REAL_INDEX])

    difference = abs(fake_score - real_score)

    if difference < UNCERTAIN_MARGIN:
        verdict = "UNCERTAIN"
    elif fake_score > real_score:
        verdict = "FAKE"
    else:
        verdict = "REAL"

    return {
        "verdict": verdict,
        "fake_score": fake_score,
        "real_score": real_score,
        "raw_output": raw_values.tolist(),
        "model_scores": scores.tolist()
    }


# ---------------- FILE HASH ----------------
def calculate_sha256(data):
    return hashlib.sha256(data).hexdigest()


# ---------------- IMAGE DETECTION ----------------
st.header("1. Image Detection")

image_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    key="image_upload"
)

if image_file is not None:
    image_bytes = image_file.getvalue()

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            width=400
        )

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                result = predict_image(image)

            st.subheader("Image Result")
            st.write(f"**Verdict: {result['verdict']}**")

            col1, col2 = st.columns(2)

            col1.metric(
                "REAL score",
                f"{result['real_score'] * 100:.2f}%"
            )

            col2.metric(
                "FAKE score",
                f"{result['fake_score'] * 100:.2f}%"
            )

            with st.expander("Diagnostic model output"):
                st.write("Raw output:", result["raw_output"])
                st.write("Model scores:", result["model_scores"])
                st.write(
                    f"Assumed class order: index {FAKE_INDEX} = FAKE, "
                    f"index {REAL_INDEX} = REAL"
                )

            report = {
                "file_name": image_file.name,
                "file_type": "image",
                "sha256": calculate_sha256(image_bytes),
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "verdict": result["verdict"],
                "real_score": result["real_score"],
                "fake_score": result["fake_score"],
                "raw_model_output": result["raw_output"],
                "model_scores": result["model_scores"],
                "warning": (
                    "Experimental prediction; not proof of authenticity."
                )
            }

            st.download_button(
                "Download Image Forensic Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name="deep_trace_image_report.json",
                mime="application/json"
            )

    except Exception as exc:
        st.error(f"Image analysis failed: {exc}")


# ---------------- VIDEO DETECTION ----------------
st.divider()
st.header("2. Video Detection")

video_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    key="video_upload"
)

if video_file is not None:
    video_bytes = video_file.getvalue()

    st.video(video_bytes)

    if st.button("Analyze Video", type="primary"):
        temp_path = None
        reader = None

        try:
            suffix = os.path.splitext(video_file.name)[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False
            ) as temp_file:
                temp_file.write(video_bytes)
                temp_path = temp_file.name

            with st.spinner("Analyzing video frames..."):
                reader = imageio.get_reader(
                    temp_path,
                    format="ffmpeg"
                )

                metadata = reader.get_meta_data()
                fps = float(metadata.get("fps") or 0)

                # Sample approximately every half-second.
                sample_every = (
                    max(1, int(round(fps * 0.5)))
                    if fps > 0 else 15
                )

                frame_results = []
                decoded_count = 0

                for frame_index, frame in enumerate(reader):
                    decoded_count += 1

                    if frame_index % sample_every != 0:
                        continue

                    if len(frame_results) >= 10:
                        break

                    frame_image = Image.fromarray(
                        np.asarray(frame).astype(np.uint8)
                    ).convert("RGB")

                    prediction = predict_image(frame_image)

                    frame_results.append({
                        "Frame": frame_index,
                        "Timestamp (seconds)": round(
                            frame_index / fps, 2
                        ) if fps > 0 else None,
                        "Prediction": prediction["verdict"],
                        "REAL score (%)": round(
                            prediction["real_score"] * 100, 2
                        ),
                        "FAKE score (%)": round(
                            prediction["fake_score"] * 100, 2
                        ),
                        "Score difference (%)": round(
                            abs(
                                prediction["real_score"]
                                - prediction["fake_score"]
                            ) * 100, 2
                        ),
                        "Raw model output": json.dumps(
                            prediction["raw_output"]
                        )
                    })

                if not frame_results:
                    raise ValueError(
                        "No frames could be analyzed."
                    )

            results_df = pd.DataFrame(frame_results)

            # Average scores across sampled frames.
            real_average = (
                results_df["REAL score (%)"].mean() / 100
            )
            fake_average = (
                results_df["FAKE score (%)"].mean() / 100
            )

            difference = abs(real_average - fake_average)

            if difference < UNCERTAIN_MARGIN:
                overall_verdict = "UNCERTAIN"
            elif fake_average > real_average:
                overall_verdict = "FAKE"
            else:
                overall_verdict = "REAL"

            st.subheader("Overall Video Result")
            st.write(f"**Verdict: {overall_verdict}**")

            col1, col2 = st.columns(2)

            col1.metric(
                "REAL score",
                f"{real_average * 100:.2f}%"
            )

           
