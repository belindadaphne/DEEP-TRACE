
import os
import io
import json
import hashlib
import tempfile
import subprocess
from datetime import datetime, timezone

import streamlit as st
import torch
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image
from torchvision import transforms
import onnxruntime as ort
import librosa
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="DEEP-TRACE", page_icon="🔎", layout="wide")

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


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    loaded_model = torch.jit.load(MODEL_PATH, map_location="cpu")
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

    if abs(real_score - fake_score) < UNCERTAIN_MARGIN:
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
        "Scores are model outputs, not guaranteed calibrated probabilities "
        "or proof of authenticity."
    )


@st.cache_resource
def load_audio_model():
    token = st.secrets.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing from Streamlit Secrets.")

    model_file = hf_hub_download(
        repo_id=AUDIO_REPO_ID,
        filename=AUDIO_MODEL_FILENAME,
        token=token
    )
    return ort.InferenceSession(
        model_file, providers=["CPUExecutionProvider"]
    )


def predict_audio_file(audio_path):
    audio, _ = librosa.load(
        audio_path, sr=AUDIO_SAMPLE_RATE, mono=True
    )
    if audio.size == 0:
        raise ValueError("No audio samples were found.")

    duration = len(audio) / AUDIO_SAMPLE_RATE

    if len(audio) > AUDIO_MAX_LENGTH:
        audio = audio[:AUDIO_MAX_LENGTH]
    elif len(audio) < AUDIO_MAX_LENGTH:
        audio = np.pad(audio, (0, AUDIO_MAX_LENGTH - len(audio)))

    audio = (audio - np.mean(audio)) / np.sqrt(np.var(audio) + 1e-5)
    audio = audio.astype(np.float32).reshape(1, AUDIO_MAX_LENGTH)

    session = load_audio_model()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: audio})
    logits = np.asarray(outputs[0]).reshape(1, -1)

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    if probs.shape[1] <= AUDIO_FAKE_INDEX:
        raise ValueError(f"Unexpected audio model output: {probs.shape}")

    fake_score = float(probs[0, AUDIO_FAKE_INDEX])
    real_score = 1.0 - fake_score

    if abs(real_score - fake_score) < UNCERTAIN_MARGIN:
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


def extract_audio_from_video(video_path, audio_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
        "-c:a", "pcm_s16le", audio_path
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if completed.returncode != 0:
        details = completed.stderr[-1200:] if completed.stderr else "Unknown FFmpeg error"
        raise RuntimeError(
            "Could not extract audio. The video may not contain audio. "
            + details
        )
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("No audio track was extracted.")


mode = st.radio(
    "Choose analysis type",
    ["Image detection", "Video detection", "Audio detection"],
    horizontal=True
)


# IMAGE DETECTION
if mode == "Image detection":
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp"]
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
                    "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
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


# VIDEO DETECTION + AUDIO EXTRACTION
elif mode == "Video detection":
    uploaded = st.file_uploader(
        "Upload a video", type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded is not None:
        video_data = uploaded.getvalue()
        st.video(video_data)

        analyze_visual = st.checkbox(
            "Analyze video frames", value=True
        )
        analyze_audio = st.checkbox(
            "Extract and analyze video's audio track", value=True
        )

        if st.button("Analyze video", type="primary"):
            video_path = None
            audio_path = None
            reader = None

            report = {
                "file_name": uploaded.name,
                "file_sha256": file_hash(video_data),
                "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
                "visual_analysis": None,
                "audio_analysis": None
            }

            try:
                suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as temp_file:
                    temp_file.write(video_data)
                    video_path = temp_file.name

                if analyze_visual:
                    rows = []
                    with st.spinner("Analyzing video frames..."):
                        reader = imageio.get_reader(video_path, format="ffmpeg")
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
                            timestamp = index / fps if fps > 0 else 0.0

                            rows.append({
                                "Frame": index,
                                "Timestamp (seconds)": round(timestamp, 2),
                                "Prediction": result["verdict"],
                                "REAL score (%)": round(result["real_score"] * 100, 2),
                                "FAKE score (%)": round(result["fake_score"] * 100, 2)
                            })

                    if rows:
                        results_df = pd.DataFrame(rows)
                        real_average = results_df["REAL score (%)"].mean() / 100
                        fake_average = results_df["FAKE score (%)"].mean() / 100

                        if abs(real_average - fake_average) < UNCERTAIN_MARGIN:
                            verdict = "UNCERTAIN"
                        elif fake_average > real_average:
                            verdict = "FAKE"
                        else:
                            verdict = "REAL"

                        st.subheader("Visual analysis result")
                        show_result({
                            "verdict": verdict,
                            "real_score": real_average,
                            "fake_score": fake_average
                        })
                        st.write(f"Frames analyzed: {len(rows)}")
                        st.write(f"Video FPS: {fps:.3f}")
                        st.dataframe(results_df, use_container_width=True)

                        report["visual_analysis"] = {
                            "verdict": verdict,
                            "average_real_score": real_average,
                            "average_fake_score": fake_average,
                            "frame_results": rows
                        }
                    else:
                        st.warning("No video frames could be analyzed.")

                if analyze_audio:
                    st.subheader("Audio track analysis")
                    with st.spinner("Extracting audio and analyzing it..."):
                        fd, audio_path = tempfile.mkstemp(suffix=".wav")
                        os.close(fd)
                        os.remove(audio_path)
                        extract_audio_from_video(video_path, audio_path)
                        audio_result = predict_audio_file(audio_path)

                    st.success("Audio track extracted.")
                    show_result(audio_result)
                    st.write(
                        f"Audio duration: {audio_result['duration_seconds']} seconds"
                    )
                    st.audio(open(audio_path, "rb").read(), format="audio/wav")
                    report["audio_analysis"] = audio_result

                if report["visual_analysis"] is not None or report["audio_analysis"] is not None:
                    st.download_button(
                        "Download combined forensic report (JSON)",
                        json.dumps(report, indent=2),
                        file_name="deeptrace_video_forensic_report.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"Video analysis failed: {e}")
                st.info(
                    "The video may not have an audio track, or the audio model "
                    "could not load. Check HF_TOKEN, the model repository, and dependencies."
                )
            finally:
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass
                for path in (video_path, audio_path):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass


# SEPARATE AUDIO FILE DETECTION
else:
    st.subheader("🎙️ Audio deepfake detection")
    st.write(
        "Upload a WAV, FLAC, or MP3 file. For audio inside a video, "
        "select Video detection instead."
    )

    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["wav", "flac", "mp3"],
        key="audio_upload"
    )

    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        st.audio(audio_bytes)
        st.write(f"File: {uploaded.name}")
        st.write(f"Size: {len(audio_bytes) / 1024:.1f} KB")
        st.write(f"SHA-256: `{file_hash(audio_bytes)}`")

        if st.button("Analyze audio", type="primary"):
            audio_path = None
            try:
                suffix = os.path.splitext(uploaded.name)[1] or ".wav"
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as temp_file:
                    temp_file.write(audio_bytes)
                    audio_path = temp_file.name

                with st.spinner("Loading audio model and analyzing audio..."):
                    result = predict_audio_file(audio_path)

                show_result(result)
                report = {
                    "file_name": uploaded.name,
                    "file_sha256": file_hash(audio_bytes),
                    "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
                    "model_repo": AUDIO_REPO_ID,
                    "result": result,
                    "disclaimer": "AI screening only; not definitive proof of authenticity."
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
                    "Check HF_TOKEN in Streamlit Secrets, that best_model.onnx exists "
                    "in your Hugging Face repository, and that dependencies are installed."
                )
            finally:
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass


st.divider()
st.caption(
    "DEEP-TRACE is an AI screening tool. Predictions can be wrong "
    "and should not be treated as definitive proof."
)
