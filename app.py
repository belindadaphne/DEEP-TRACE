
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

# Audio detection dependencies
import onnxruntime as ort
import librosa
from huggingface_hub import hf_hub_download

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🔎",
    layout="wide"
)

MODEL_PATH = "deeptrace_model_scripted.pt"
AUDIO_REPO_ID = "deeptrace-research/deep-trace-audio"
AUDIO_MODEL_FILENAME = "best_model.onnx"

FAKE_INDEX = 0
REAL_INDEX = 1
AUDIO_FAKE_INDEX = 1

UNCERTAIN_MARGIN = 0.10
MAX_VIDEO_FRAMES = 10
FRAME_INTERVAL = 15
AUDIO_SAMPLE_RATE = 16000
AUDIO_MAX_LENGTH = 48000

st.title("🔎 DEEP-TRACE")
st.caption("AI-assisted deepfake screening")

# ---------------- IMAGE / VIDEO MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    loaded_model = torch.jit.load(
        MODEL_PATH, map_location="cpu"
    )
    loaded_model.eval()
    return loaded_model


try:
    model = load_model()
    st.success("Image/video detection model loaded")
except Exception as e:
    st.error(f"Could not load image/video model: {e}")
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
    col1.metric(
        "REAL score",
        f'{result["real_score"] * 100:.2f}%'
    )
    col2.metric(
        "FAKE score",
        f'{result["fake_score"] * 100:.2f}%'
    )

    st.caption(
        "Scores are model outputs, not guaranteed calibrated "
        "probabilities or proof of authenticity."
    )


# ---------------- AUDIO MODEL ----------------
@st.cache_resource
def load_audio_model():
    token = st.secrets.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it in Streamlit "
            "App settings > Secrets."
        )

    model_file = hf_hub_download(
        repo_id=AUDIO_REPO_ID,
        filename=AUDIO_MODEL_FILENAME,
        token=token
    )

    session = ort.InferenceSession(
        model_file,
        providers=["CPUExecutionProvider"]
    )

    return session


def predict_audio(audio_bytes, filename):
    temp_path = None

    try:
        suffix = os.path.splitext(filename)[1] or ".wav"

        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        audio, sr = librosa.load(
            temp_path,
            sr=AUDIO_SAMPLE_RATE,
            mono=True
        )

        if len(audio) == 0:
            raise ValueError("The audio file is empty.")

        duration = len(audio) / AUDIO_SAMPLE_RATE

        # Model expects 3 seconds at 16 kHz
        if len(audio) > AUDIO_MAX_LENGTH:
            audio = audio[:AUDIO_MAX_LENGTH]
        elif len(audio) < AUDIO_MAX_LENGTH:
            audio = np.pad(
                audio,
                (0, AUDIO_MAX_LENGTH - len(audio))
            )

        # Match the model card's preprocessing
        audio = (
            audio - np.mean(audio)
        ) / np.sqrt(np.var(audio) + 1e-5)

        audio = audio.astype(np.float32).reshape(
            1, AUDIO_MAX_LENGTH
        )

        session = load_audio_model()
        input_name = session.get_inputs()[0].name

        outputs = session.run(
            None, {input_name: audio}
        )

        logits = np.asarray(outputs[0]).reshape(1, -1)

        # Convert logits to probabilities safely
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(
            exp_scores, axis=1, keepdims=True
        )

        if probs.shape[1] <= AUDIO_FAKE_INDEX:
            raise ValueError(
                "Unexpected output shape from the audio model."
            )

        fake_score = float(probs[0, AUDIO_FAKE_INDEX])
        real_score = 1.0 - fake_score

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
            "fake_score": fake_score,
            "duration_seconds": round(duration, 2),
            "sample_rate": AUDIO_SAMPLE_RATE
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------- INPUT ----------------
mode = st.radio(
    "Choose analysis type",
    [
        "Image detection",
        "Video detection",
        "Audio detection"
    ],
    horizontal=True
)


# ---------------- IMAGE DETECTION ----------------
if mode == "Image detection":
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded is not None:
        data = uploaded.getvalue()
        image = Image.open(io.BytesIO(data))
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

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


# ---------------- VIDEO DETECTION ----------------
elif mode == "Video detection":
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
                    suffix=os.path.splitext(uploaded.name)[1]
                    or ".mp4",
                    delete=False
                ) as temp_file:
                    temp_file.write(video_data)
                    temp_path = temp_file.name

                rows = []

                with st.spinner(
                    "Extracting and analyzing video frames..."
                ):
                    reader = imageio.get_reader(
                        temp_path, format="ffmpeg"
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
                            "Timestamp (seconds)": round(
                                timestamp, 2
                            ),
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

                    difference = abs(
                        real_average - fake_average
                    )

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


# ---------------- AUDIO DETECTION ----------------
else:
    st.subheader("🎙️ Audio deepfake detection")
    st.write(
        "Upload a WAV, FLAC, or MP3 file to screen it "
        "with the audio model."
    )

    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["wav", "flac", "mp3"],
        key="audio_upload"
    )

    if uploaded is not None:
        audio_bytes = uploaded.getvalue()

        st.audio(audio_bytes)

        st.write(f"**File:** {uploaded.name}")
        st.write(f"**Size:** {len(audio_bytes) / 1024:.1f} KB")
        st.write(f"**SHA-256:** `{file_hash(audio_bytes)}`")

        if st.button("Analyze audio", type="primary"):
            try:
                with st.spinner(
                    "Loading audio model and analyzing audio..."
                ):
                    result = predict_audio(
                        audio_bytes, uploaded.name
                    )

                st.subheader("Audio analysis result")
                show_result(result)

                st.write(
                    f"Duration: {result['duration_seconds']} seconds"
                )
                st.write(
                    f"Model sample rate: "
                    f"{result['sample_rate']} Hz"
                )

                report = {
                    "file_name": uploaded.name,
                    "file_sha256": file_hash(audio_bytes),
                    "analysis_time_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "model_repo": AUDIO_REPO_ID,
                    "result": result,
                    "disclaimer": (
                        "AI screening only; not definitive proof "
                        "of authenticity."
                    )
                }

                st.download_button(
                    "Download audio forensic report (JSON)",
                    json.dumps(report, indent=2),
                    file_name="deeptrace_audio_report.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Audio analysis failed: {e}")
                st.info(
                    "Check that HF_TOKEN is configured in "
                    "Streamlit Secrets, the model file exists "
                    "in your Hugging Face repository, and all "
                    "audio dependencies are installed."
                )


# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "DEEP-TRACE is an AI screening tool. Its predictions "
    "can be wrong and should not be treated as definitive proof."
)
