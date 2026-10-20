import io
import os
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
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model


try:
    model = load_model()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_image(image):
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise ValueError("Unsupported model output.")

    output = output.detach().float().cpu()

    if output.ndim == 2:
        output = output[0]

    if output.ndim != 1 or output.numel() != 2:
        raise ValueError(f"Unexpected output shape: {tuple(output.shape)}")

    raw = output.numpy().astype(float)

    if (
        np.all(raw >= 0)
        and np.all(raw <= 1)
        and np.isclose(raw.sum(), 1.0, atol=1e-3)
    ):
        scores = raw
    else:
        exp_values = np.exp(raw - np.max(raw))
        scores = exp_values / exp_values.sum()

    fake_score = float(scores[FAKE_INDEX])
    real_score = float(scores[REAL_INDEX])

    if abs(fake_score - real_score) < UNCERTAIN_MARGIN:
        verdict = "UNCERTAIN"
    elif fake_score > real_score:
        verdict = "FAKE"
    else:
        verdict = "REAL"

    return {
        "verdict": verdict,
        "fake_score": fake_score,
        "real_score": real_score,
        "raw_output": raw.tolist(),
    }


def file_hash(data):
    return hashlib.sha256(data).hexdigest()


def download_json(report, filename, label):
    st.download_button(
        label,
        data=json.dumps(report, indent=2),
        file_name=filename,
        mime="application/json",
    )


# IMAGE DETECTION
st.header("1. Image Detection")

image_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    key="image_upload",
)

if image_file is not None:
    image_bytes = image_file.getvalue()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(image, caption="Uploaded image", width=360)

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                result = predict_image(image)

            st.subheader(f"Image verdict: {result['verdict']}")

            col1, col2 = st.columns(2)
            col1.metric("REAL score", f"{result['real_score'] * 100:.2f}%")
            col2.metric("FAKE score", f"{result['fake_score'] * 100:.2f}%")

            with st.expander("Diagnostic model output"):
                st.write("Raw output:", result["raw_output"])
                st.caption(
                    "Class order assumes index 0 = FAKE and index 1 = REAL. "
                    "Verify this against the training code."
                )

            download_json(
                {
                    "file_name": image_file.name,
                    "type": "image",
                    "sha256": file_hash(image_bytes),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    **result,
                },
                "deep_trace_image_report.json",
                "Download image report (JSON)",
            )

    except Exception as exc:
        st.error(f"Image analysis failed: {exc}")


# VIDEO DETECTION
st.divider()
st.header("2. Video Detection")

video_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    key="video_upload",
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
                suffix=suffix, delete=False
            ) as temp_file:
                temp_file.write(video_bytes)
                temp_path = temp_file.name

            reader = imageio.get_reader(temp_path, format="ffmpeg")
            metadata = reader.get_meta_data()
            fps = float(metadata.get("fps") or 0)

            every_n_frames = (
                max(1, round(fps * 0.5)) if fps > 0 else 15
            )
            frame_rows = []

            with st.spinner("Analyzing video frames..."):
                for frame_index, frame in enumerate(reader):
                    if frame_index % every_n_frames != 0:
                        continue

                    frame_image = Image.fromarray(
                        np.asarray(frame).astype(np.uint8)
                    ).convert("RGB")

                    result = predict_image(frame_image)

                    frame_rows.append({
                        "Frame": frame_index,
                        "Timestamp (seconds)": (
                            round(frame_index / fps, 2) if fps > 0 else None
                        ),
                        "Prediction": result["verdict"],
                        "REAL score (%)": round(
                            result["real_score"] * 100, 2
                        ),
                        "FAKE score (%)": round(
                            result["fake_score"] * 100, 2
                        ),
                    })

                    if len(frame_rows) >= 10:
                        break

            if not frame_rows:
                raise ValueError("No video frames could be decoded.")

            results_df = pd.DataFrame(frame_rows)
            real_average = float(results_df["REAL score (%)"].mean() / 100)
            fake_average = float(results_df["FAKE score (%)"].mean() / 100)

            if abs(real_average - fake_average) < UNCERTAIN_MARGIN:
                verdict = "UNCERTAIN"
            elif fake_average > real_average:
                verdict = "FAKE"
            else:
                verdict = "REAL"

            st.subheader(f"Overall video verdict: {verdict}")

            col1, col2 = st.columns(2)
            col1.metric("Average REAL score", f"{real_average * 100:.2f}%")
            col2.metric("Average FAKE score", f"{fake_average * 100:.2f}%")

            st.write(f"Frames analyzed: {len(frame_rows)}")
            st.write(f"Video FPS: {fps:.3f}" if fps > 0 else "FPS unavailable")

            st.subheader("Sampled frame results")
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

            video_hash = file_hash(video_bytes)
            st.subheader("File integrity (SHA-256)")
            st.code(video_hash)

            download_json(
                {
                    "file_name": video_file.name,
                    "type": "video",
                    "sha256": video_hash,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "fps": fps,
                    "frames_analyzed": len(frame_rows),
                    "overall_verdict": verdict,
                    "average_real_score": real_average,
                    "average_fake_score": fake_average,
                    "frame_results": frame_rows,
                },
                "deep_trace_video_report.json",
                "Download forensic report (JSON)",
            )

            st.download_button(
                "Download frame results (CSV)",
                data=results_df.to_csv(index=False),
                file_name="deep_trace_frame_results.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")

        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


# AUDIO SECTION
st.divider()
st.header("3. Audio Detection")

st.warning(
    "Audio upload is available, but no trained audio deepfake model "
    "is configured. This section does not classify audio as REAL or FAKE."
)

audio_file = st.file_uploader(
    "Upload audio",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
    key="audio_upload",
)

if audio_file is not None:
    audio_bytes = audio_file.getvalue()
    st.audio(audio_bytes)

    st.write(f"File: {audio_file.name}")
    st.write(f"Size: {len
