
import os
import io
import json
import hashlib
import tempfile
from datetime import datetime

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms


# ================ CONFIGURATION ================

st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🔎",
    layout="wide"
)

MODEL_PATH = "deeptrace_model_scripted.pt"
UNCERTAIN_THRESHOLD = 0.10
FRAME_INTERVAL = 15
MAX_VIDEO_FRAMES = 300


# ================ STYLING ================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080808, #190707);
    color: white;
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ================ HEADER ================

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


# ================ LOAD MODEL ================

@st.cache_resource
def load_model():
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model


try:
    model = load_model()
    st.success("Detection model loaded.")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# ================ PREPROCESSING ================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================ PREDICTION ================

def sha256_hash(data):
    return hashlib.sha256(data).hexdigest()


def predict_image(image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        output = output.reshape(-1)

        if output.numel() != 2:
            raise ValueError(
                f"Expected two model outputs, got {output.numel()}."
            )

        scores = torch.softmax(output, dim=0).cpu().numpy()

    # DEBUG: inspect model outputs in Streamlit.
    st.write("DEBUG — raw model output:", output.tolist())
    st.write("DEBUG — model scores:", scores.tolist())

    # Assumption: output index 0 = fake, index 1 = real.
    # This mapping still needs verification against training code.
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
        st.warning("Verdict: UNCERTAIN — scores are close.")

    col1, col2 = st.columns(2)
    col1.metric("REAL score", f"{result['real_score'] * 100:.2f}%")
    col2.metric("FAKE score", f"{result['fake_score'] * 100:.2f}%")

    st.caption(
        "Model scores are not guaranteed calibrated probabilities "
        "or measures of detection accuracy."
    )


def report_json(data):
    return json.dumps(data, indent=2, ensure_ascii=False)


# ================ MEDIA SELECTION ================

media_type = st.radio(
    "Choose media type",
    ["Image", "Video"],
    horizontal=True
)


# ================ IMAGE DETECTION ================

if media_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True
            )

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
                    "disclaimer": "AI prediction, not conclusive proof."
                }

                st.subheader("File integrity")
                st.code(report["sha256"])

                st.download_button(
                    "Download forensic report (JSON)",
                    data=report_json(report),
                    file_name="deep_trace_image_report.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"Could not analyze image: {e}")


# ================ VIDEO DETECTION ================

else:
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv", "webm"]
    )

    if uploaded_video is not None:
        video_bytes = uploaded_video.getvalue()
        st.video(video_bytes)

        if st.button("Analyze video", type="primary"):
            temp_video_path = None
            reader = None

            try:
                suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False
                ) as temp_file:
                    temp_file.write(video_bytes)
                    temp_video_path = temp_file.name

                frame_results = []
                fps = None
                frames_decoded = 0

                with st.spinner("Analyzing sampled video frames..."):
                    reader = imageio.get_reader(
                        temp_video_path,
                        format="ffmpeg"
                    )

                    metadata = reader.get_meta_data()

                    try:
                        candidate_fps = float(metadata.get("fps"))
                        if np.isfinite(candidate_fps) and candidate_fps > 0:
                            fps = candidate_fps
                    except (TypeError, ValueError):
                        fps = None

                    for index, frame in enumerate(reader):
                        frames_decoded += 1

                        if index % FRAME_INTERVAL != 0:
                            continue

                        if len(frame_results) >= MAX_VIDEO_FRAMES:
                            break

                        result = predict_image(Image.fromarray(frame))

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

                if not frame_results:
                    st.error("No video frames could be analyzed.")
                else:
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

                    st.write(f"**Frames analyzed:** {len(frame_results)}")
                    st.write(f"**Frames decoded:** {frames_decoded}")
                    st.write(
                        f"**Video FPS:** {fps:.3f}"
                        if fps is not None
                        else "**Video FPS:** Unavailable"
                    )

                    st.subheader("Sampled frame results")

                    display_df = results_df.copy()
                    display_df["real_score"] *= 100
                    display_df["fake_score"] *= 100
                    display_df["score_difference"] *= 100

                    display_df = display_df.rename(columns={
                        "frame": "Frame",
                        "timestamp_seconds": "Timestamp (seconds)",
                        "prediction": "Prediction",
                        "real_score": "REAL score (%)",
                        "fake_score": "FAKE score (%)",
                        "score_difference": "Score difference (%)"
                    })

                    st.dataframe(
                        display_df.round(2),
                        use_container_width=True
                    )

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
                        "frames_decoded": frames_decoded,
                        "frames_analyzed": len(frame_results),
                        "frame_interval": FRAME_INTERVAL,
                        "frame_results": frame_results,
                        "disclaimer": (
                            "The result aggregates sampled frames; "
                            "it is not conclusive proof of authenticity."
                        )
                    }

                    st.subheader("File integrity")
                    st.code(report["sha256"])

                    st.download_button(
                        "Download forensic report (JSON)",
                        data=report_json(report),
                        file_name="deep_trace_video_report.json",
                        mime="application/json"
                    )

                    st.download_button(
                        "Download frame results (CSV)",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="deep_trace_video_frames.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Could not analyze video: {e}")
                st.info("Try an MP4 video encoded with H.264.")

            finally:
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass

                if (
                    temp_video_path is not None
                    and os.path.exists(temp_video_path)
                ):
                    try:
                        os.remove(temp_video_path)
                    except OSError:
                        pass
