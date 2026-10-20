
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

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🔎",
    layout="wide"
)

MODEL_PATH = "deeptrace_model_scripted.pt"
FAKE_INDEX = 0
REAL_INDEX = 1
UNCERTAIN_MARGIN = 0.10
MAX_VIDEO_FRAMES = 10
FRAME_INTERVAL = 15

st.title("🔎 DEEP-TRACE")
st.caption("AI-assisted deepfake screening")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

try:
    model = load_model()
    st.success("Detection model loaded")
except Exception as e:
    st.error(f"Could not load model: {e}")
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
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        output = output.reshape(-1)

        if output.numel() == 1:
            fake_score = torch.sigmoid(output[0]).item()
            real_score = 1.0 - fake_score
        else:
            scores = torch.softmax(output, dim=0)
            fake_score = float(scores[FAKE_INDEX].item())
            real_score = float(scores[REAL_INDEX].item())

    difference = abs(real_score - fake_score)

    if difference < UNCERTAIN_MARGIN:
        verdict = "UNCERTAIN"
    elif fake_score > real_score:
        verdict = "FAKE"
    else:
        verdict = "REAL"

    return {
        "verdict": verdict,
        "real_score": real_score,
        "fake_score": fake_score
    }


def file_hash(data):
    return hashlib.sha256(data).hexdigest()


def show_result(result):
    verdict = result["verdict"]

    if verdict == "FAKE":
        st.error("Verdict: FAKE")
    elif verdict == "REAL":
        st.success("Verdict: REAL")
    else:
        st.warning("Verdict: UNCERTAIN")

    col1, col2 = st.columns(2)
    col1.metric("REAL score", f'{result["real_score"] * 100:.2f}%')
    col2.metric("FAKE score", f'{result["fake_score"] * 100:.2f}%')

    st.caption(
        "Scores are model outputs, not guaranteed calibrated "
        "probabilities or proof of authenticity."
    )


# ---------------- INPUT ----------------
mode = st.radio(
    "Choose analysis type",
    ["Image detection", "Video detection"],
    horizontal=True
)

if mode == "Image detection":
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded is not None:
        data = uploaded.getvalue()
        image = Image.open(io.BytesIO(data))
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Analyze image", type="primary"):
            try:
                with st.spinner("Analyzing image..."):
                    result = predict_image(image)

                show_result(result)

                report = {
                    "file_name": uploaded.name,
                    "file_sha256": file_hash(data),
                    "analysis_time_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "result": result
                }

                st.download_button(
                    "Download forensic report (JSON)",
                    json.dumps(report, indent=2),
                    file_name="deeptrace_image_report.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Image analysis failed: {e}")


else:
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded is not None:
        video_data = uploaded.getvalue()
        st.video(video_data)

        if st.button("Analyze video", type="primary"):
            temp_path = None
            reader = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(uploaded.name)[1] or ".mp4",
                    delete=False
                ) as temp_file:
                    temp_file.write(video_data)
                    temp_path = temp_file.name

                rows = []

                with st.spinner("Extracting and analyzing video frames..."):
                    reader = imageio.get_reader(
                        temp_path,
                        format="ffmpeg"
                    )

                    metadata = reader.get_meta_data()
                    fps = float(metadata.get("fps") or 0)

                    for index, frame in enumerate(reader):
                        if index % FRAME_INTERVAL != 0:
                            continue

                        if len(rows) >= MAX_VIDEO_FRAMES:
                            break

                        image = Image.fromarray(
                            np.asarray(frame).astype(np.uint8)
                        )

                        result = predict_image(image)

                        timestamp = (
                            index / fps if fps > 0 else 0.0
                        )

                        rows.append({
                            "Frame": index,
                            "Timestamp (seconds)": round(timestamp, 2),
                            "Prediction": result["verdict"],
                            "REAL score (%)": round(
                                result["real_score"] * 100, 2
                            ),
                            "FAKE score (%)": round(
                                result["fake_score"] * 100, 2
                            ),
                            "Score difference (%)": round(
                                abs(
                                    result["real_score"]
                                    - result["fake_score"]
                                ) * 100, 2
                            )
                        })

                if not rows:
                    st.error(
                        "No frames could be extracted. "
                        "Try another video in MP4 format."
                    )
                else:
                    results_df = pd.DataFrame(rows)

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

                    st.subheader("Overall video result")
                    show_result({
                        "verdict": overall_verdict,
                        "real_score": real_average,
                        "fake_score": fake_average
                    })

                    st.write(f"Frames analyzed: {len(rows)}")
                    st.write(f"Video FPS: {fps:.3f}")

                    st.subheader("Sampled frame results")
                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )

                    report = {
                        "file_name": uploaded.name,
                        "file_sha256": file_hash(video_data),
                        "analysis_time_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "frames_analyzed": len(rows),
                        "fps": fps,
                        "overall_verdict": overall_verdict,
                        "average_real_score": real_average,
                        "average_fake_score": fake_average,
                        "frame_results": rows
                    }

                    st.subheader("File integrity")
                    st.code(file_hash(video_data))

                    st.download_button(
                        "Download forensic report (JSON)",
                        json.dumps(report, indent=2),
                        file_name="deeptrace_video_report.json",
                        mime="application/json"
                    )

                    st.download_button(
                        "Download frame results (CSV)",
                        results_df.to_csv(index=False),
                        file_name="deeptrace_frame_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Video analysis failed: {e}")
                st.info(
                    "If video decoding fails, try converting "
                    "the video to MP4 and uploading it again."
                )

            finally:
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass

                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

st.divider()
st.caption(
    "DEEP-TRACE is an AI screening tool. Its predictions can be "
    "wrong and should not be treated as definitive proof."
)
