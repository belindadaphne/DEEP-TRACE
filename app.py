
import streamlit as st
import torch
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import hashlib
import json
import io
from datetime import datetime
from PIL import Image
from torchvision import transforms

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🔎",
    layout="wide"
)

MODEL_PATH = "deeptrace_model_scripted.pt"

# Lower threshold: fewer UNCERTAIN verdicts
UNCERTAIN_THRESHOLD = 0.10

MAX_VIDEO_FRAMES = 300
FRAME_INTERVAL = 15

# ---------------- STYLING ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080808, #190707);
    color: white;
}
h1, h2, h3 {
    color: white;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    margin: 12px 0;
    background: #171717;
    border: 1px solid #444;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    "<h1>DEEP<span style='color:#ef1725'>-</span>TRACE</h1>",
    unsafe_allow_html=True
)
st.markdown("### AI-POWERED MEDIA AUTHENTICITY ANALYSIS")
st.markdown("## Is it REAL or FAKE?")
st.write(
    "Analyze images and videos using deep learning. "
    "Review sampled frame predictions and download forensic reports."
)
st.divider()

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

try:
    model = load_model()
    st.success("Detection model loaded.")
except Exception as e:
    st.error(f"Could not load the detection model: {e}")
    st.stop()

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- UTILITIES ----------------
def sha256_hash(data):
    return hashlib.sha256(data).hexdigest()


def predict_image(image):
    """Return verdict and model scores for one image."""
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        output = output.reshape(-1)

        # Convert model output to scores.
        scores = torch.softmax(output, dim=0).cpu().numpy()

    fake_score = float(scores[0])
    real_score = float(scores[1])

    difference = abs(fake_score - real_score)

    if difference < UNCERTAIN_THRESHOLD:
        verdict = "UNCERTAIN"
    elif fake_score > real_score:
        verdict = "FAKE"
    else:
        verdict = "REAL"

    return {
        "verdict": verdict,
        "fake_score": fake_score,
        "real_score": real_score,
        "difference": difference
    }


def display_result(result):
    verdict = result["verdict"]

    if verdict == "REAL":
        st.success("Verdict: REAL")
    elif verdict == "FAKE":
        st.error("Verdict: FAKE")
    else:
        st.warning("Verdict: UNCERTAIN — the scores are close.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("REAL score", f"{result['real_score'] * 100:.2f}%")

    with col2:
        st.metric("FAKE score", f"{result['fake_score'] * 100:.2f}%")

    st.caption(
        "Scores are model outputs converted to relative scores; "
        "they are not guaranteed to be calibrated probabilities or accuracy."
    )


def create_report(report_data):
    return json.dumps(report_data, indent=2, ensure_ascii=False)


# ---------------- MEDIA TYPE ----------------
media_type = st.radio(
    "Choose media type",
    ["Image", "Video"],
    horizontal=True
)

# ---------------- IMAGE DETECTION ----------------
if media_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a JPG, JPEG, PNG, or WEBP image."
    )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

            st.image(image, caption="Uploaded image", use_container_width=True)

            if st.button("Analyze image", type="primary"):
                with st.spinner("Analyzing image..."):
                    result = predict_image(image)

                st.subheader("Detection result")
                display_result(result)

                report = {
                    "application": "DEEP-TRACE",
                    "media_type": "Image",
                    "filename": uploaded_file.name,
                    "analysis_time": datetime.now().isoformat(),
                    "sha256": sha256_hash(file_bytes),
                    "verdict": result["verdict"],
                    "real_score": result["real_score"],
                    "fake_score": result["fake_score"],
                    "uncertain_threshold": UNCERTAIN_THRESHOLD,
                    "disclaimer": (
                        "AI prediction only. Results are not conclusive proof "
                        "of authenticity or manipulation."
                    )
                }

                st.markdown("### File integrity")
                st.code(report["sha256"])

                st.download_button(
                    "Download forensic report (JSON)",
                    data=create_report(report),
                    file_name="deep_trace_image_report.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"Could not analyze this image: {e}")


# ---------------- VIDEO DETECTION ----------------
else:
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="Upload a supported video file."
    )

    if uploaded_video is not None:
        video_bytes = uploaded_video.getvalue()

        st.video(video_bytes)

        if st.button("Analyze video", type="primary"):
            temp_video = io.BytesIO(video_bytes)
            frame_results = []
            fps = None
            frame_count = 0
            sampled_count = 0

            try:
                reader = imageio.get_reader(temp_video, format="ffmpeg")
                metadata = reader.get_meta_data()

                raw_fps = metadata.get("fps")
                if raw_fps is not None:
                    try:
                        raw_fps = float(raw_fps)
                        if np.isfinite(raw_fps) and raw_fps > 0:
                            fps = raw_fps
                    except (TypeError, ValueError):
                        fps = None

                with st.spinner("Analyzing sampled video frames..."):
                    for index, frame in enumerate(reader):
                        frame_count += 1

                        if index % FRAME_INTERVAL != 0:
                            continue

                        if sampled_count >= MAX_VIDEO_FRAMES:
                            break

                        pil_frame = Image.fromarray(frame).convert("RGB")
                        result = predict_image(pil_frame)

                        timestamp = (
                            round(index / fps, 3)
                            if fps is not None
                            else None
                        )

                        frame_results.append({
                            "frame": index,
                            "timestamp_seconds": timestamp,
                            "prediction": result["verdict"],
                            "real_score": round(result["real_score"], 6),
                            "fake_score": round(result["fake_score"], 6),
                            "score_difference": round(
                                result["difference"], 6
                            )
                        })

                        sampled_count += 1

                reader.close()

                if not frame_results:
                    st.error("No video frames could be analyzed.")
                    st.stop()

                results_df = pd.DataFrame(frame_results)

                mean_real = float(results_df["real_score"].mean())
                mean_fake = float(results_df["fake_score"].mean())
                difference = abs(mean_real - mean_fake)

                if difference < UNCERTAIN_THRESHOLD:
                    final_verdict = "UNCERTAIN"
                elif mean_fake > mean_real:
                    final_verdict = "FAKE"
                else:
                    final_verdict = "REAL"

                st.subheader("Overall video result")

                display_result({
                    "verdict": final_verdict,
                    "real_score": mean_real,
                    "fake_score": mean_fake,
                    "difference": difference
                })

                st.write(f"**Frames analyzed:** {sampled_count}")

                if fps is not None:
                    st.write(f"**Video FPS:** {fps:.3f}")
                else:
                    st.write("**Video FPS:** Unavailable")

                if frame_count > 0:
                    st.write(f"**Frames decoded:** {frame_count}")

                st.subheader("Sampled frame results")

                display_df = results_df.copy()
                display_df["real_score"] = (
                    display_df["real_score"] * 100
                ).round(2)
                display_df["fake_score"] = (
                    display_df["fake_score"] * 100
                ).round(2)
                display_df["score_difference"] = (
                    display_df["score_difference"] * 100
                ).round(2)

                display_df = display_df.rename(columns={
                    "frame": "Frame",
                    "timestamp_seconds": "Timestamp (seconds)",
                    "prediction": "Prediction",
                    "real_score": "REAL score (%)",
                    "fake_score": "FAKE score (%)",
                    "score_difference": "Score difference (%)"
                })

                st.dataframe(display_df, use_container_width=True)

                report = {
                    "application": "DEEP-TRACE",
                    "media_type": "Video",
                    "filename": uploaded_video.name,
                    "analysis_time": datetime.now().isoformat(),
                    "sha256": sha256_hash(video_bytes),
                    "overall_verdict": final_verdict,
                    "mean_real_score": mean_real,
                    "mean_fake_score": mean_fake,
                    "uncertain_threshold": UNCERTAIN_THRESHOLD,
                    "video_fps": fps,
                    "frames_decoded": frame_count,
                    "frames_analyzed": sampled_count,
                    "frame_interval": FRAME_INTERVAL,
                    "frame_results": frame_results,
                    "disclaimer": (
                        "The overall result aggregates sampled frame scores. "
                        "It does not establish that the entire video is real "
                        "or fake. AI predictions are not conclusive proof."
                    )
                }

                st.markdown("### File integrity")
                st.code(report["sha256"])

                st.download_button(
                    "Download forensic report (JSON)",
                    data=create_report(report),
                    file_name="deep_trace_video_report.json",
                    mime="application/json"
                )

                csv_data = results_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download frame results (CSV)",
                    data=csv_data,
                    file_name="deep_trace_video_frames.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Could not analyze this video: {e}")
                st.info(
                    "Try an MP4 video encoded with H.264 if the file "
                    "cannot be decoded."
                )
