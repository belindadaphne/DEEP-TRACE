
import os
import json
import tempfile
import subprocess

import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import imageio_ffmpeg
import librosa
import onnxruntime as ort
from huggingface_hub import hf_hub_download


# ---------------- CONFIG ----------------

AUDIO_REPO_ID = "deeptrace-research/deep-trace-audio"
AUDIO_MODEL_FILENAME = "best_model.onnx"

FAKE_INDEX = 0
REAL_INDEX = 1
AUDIO_FAKE_INDEX = 1

UNCERTAIN_MARGIN = 0.10
MAX_VIDEO_FRAMES = 10
FRAME_INTERVAL = 15

AUDIO_SAMPLE_RATE = 16000
AUDIO_MAX_LENGTH = 48000  # 3 seconds


# ---------------- PAGE ----------------

st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 DEEP-TRACE")
st.write("AI-assisted image, video, and audio authenticity analysis.")
st.warning(
    "Predictions are estimates, not proof that media is real or fake. "
    "Low-confidence results should be treated as inconclusive."
)


# ---------------- IMAGE MODEL ----------------

@st.cache_resource
def load_model():
    model_path = "deeptrace_model_scripted.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing {model_path}. Add the model file to your repository."
        )

    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def classify_scores(real_score, fake_score):
    difference = abs(float(real_score) - float(fake_score))

    if difference < UNCERTAIN_MARGIN:
        label = "UNCERTAIN"
    elif fake_score > real_score:
        label = "FAKE"
    else:
        label = "REAL"

    return label


def predict_image(image):
    model = load_model()

    image = image.convert("RGB")
    tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        if output.numel() == 1:
            fake_score = torch.sigmoid(output).item()
            real_score = 1.0 - fake_score
        else:
            probabilities = F.softmax(output, dim=1)[0]
            fake_score = probabilities[FAKE_INDEX].item()
            real_score = probabilities[REAL_INDEX].item()

    label = classify_scores(real_score, fake_score)

    return {
        "label": label,
        "real_score": round(real_score * 100, 2),
        "fake_score": round(fake_score * 100, 2)
    }


# ---------------- AUDIO MODEL ----------------

@st.cache_resource
def load_audio_model():
    token = st.secrets.get("HF_TOKEN", None)

    model_path = hf_hub_download(
        repo_id=AUDIO_REPO_ID,
        filename=AUDIO_MODEL_FILENAME,
        token=token
    )

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    return session


def predict_audio_file(audio_path):
    session = load_audio_model()

    audio, _ = librosa.load(
        audio_path,
        sr=AUDIO_SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    if len(audio) < AUDIO_MAX_LENGTH:
        audio = np.pad(
            audio,
            (0, AUDIO_MAX_LENGTH - len(audio))
        )
    else:
        audio = audio[:AUDIO_MAX_LENGTH]

    audio = audio.astype(np.float32)

    # Match the normalization used by the existing app.
    audio = (audio - audio.mean()) / (
        np.sqrt(audio.var() + 1e-5)
    )

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    model_input = audio

    # Adapt the input dimensions to the ONNX model.
    if len(input_shape) == 2:
        model_input = model_input[np.newaxis, :]
    elif len(input_shape) == 3:
        model_input = model_input[np.newaxis, np.newaxis, :]

    outputs = session.run(
        None,
        {input_name: model_input.astype(np.float32)}
    )

    logits = np.asarray(outputs[0])

    if logits.size == 1:
        fake_score = float(1 / (1 + np.exp(-logits.reshape(-1)[0])))
        real_score = 1.0 - fake_score
    else:
        logits = logits.reshape(1, -1)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        fake_score = float(probabilities[0][AUDIO_FAKE_INDEX])
        real_index = 1 - AUDIO_FAKE_INDEX
        real_score = float(probabilities[0][real_index])

    return {
        "label": classify_scores(real_score, fake_score),
        "real_score": round(real_score * 100, 2),
        "fake_score": round(fake_score * 100, 2)
    }


# ---------------- VIDEO AUDIO EXTRACTION ----------------

def extract_audio_from_video(video_path, audio_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", video_path,
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        audio_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180
    )

    if result.returncode != 0 or not os.path.exists(audio_path):
        message = result.stderr[-1200:] if result.stderr else (
            "No audio stream was found."
        )
        raise RuntimeError(message)

    if os.path.getsize(audio_path) == 0:
        raise RuntimeError("The extracted audio file is empty.")


# ---------------- IMAGE TAB ----------------

image_tab, video_tab, audio_tab = st.tabs([
    "🖼️ Image Detection",
    "🎬 Video Detection",
    "🎙️ Audio Detection"
])


with image_tab:
    st.subheader("Analyze an image")

    uploaded_image = st.file_uploader(
        "Upload JPG, JPEG, or PNG",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image", width=400)

        if st.button("Analyze image", key="analyze_image"):
            try:
                with st.spinner("Analyzing image..."):
                    result = predict_image(image)

                st.subheader(f"Result: {result['label']}")
                st.metric("Fake score", f"{result['fake_score']}%")
                st.metric("Real score", f"{result['real_score']}%")

            except Exception as error:
                st.error(f"Image analysis failed: {error}")


# ---------------- VIDEO TAB ----------------

with video_tab:
    st.subheader("Analyze a video")

    uploaded_video = st.file_uploader(
        "Upload MP4, MOV, or AVI",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload"
    )

    analyze_frames = st.checkbox(
        "Analyze video frames",
        value=True
    )

    analyze_video_audio = st.checkbox(
        "Extract and analyze video's audio",
        value=False
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Analyze video", key="analyze_video"):
            video_path = None
            audio_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp_video:
                    temp_video.write(uploaded_video.getbuffer())
                    video_path = temp_video.name

                frame_results = []

                if analyze_frames:
                    with st.spinner("Analyzing video frames..."):
                        reader = imageio.get_reader(video_path)

                        try:
                            total_frames = reader.count_frames()
                        except Exception:
                            total_frames = MAX_VIDEO_FRAMES * FRAME_INTERVAL

                        frame_indices = list(
                            range(0, total_frames, FRAME_INTERVAL)
                        )[:MAX_VIDEO_FRAMES]

                        for frame_number in frame_indices:
                            try:
                                frame = reader.get_data(frame_number)
                                frame_image = Image.fromarray(
                                    np.asarray(frame).astype(np.uint8)
                                ).convert("RGB")

                                prediction = predict_image(frame_image)

                                frame_results.append({
                                    "Frame": frame_number,
                                    "Prediction": prediction["label"],
                                    "Real score (%)": prediction["real_score"],
                                    "Fake score (%)": prediction["fake_score"]
                                })

                            except Exception as frame_error:
                                frame_results.append({
                                    "Frame": frame_number,
                                    "Prediction": "ERROR",
                                    "Real score (%)": None,
                                    "Fake score (%)": None,
                                    "Error": str(frame_error)
                                })

                        reader.close()

                    if frame_results:
                        results_df = pd.DataFrame(frame_results)
                        st.subheader("Frame-by-frame results")
                        st.dataframe(results_df, use_container_width=True)

                        valid_rows = results_df[
                            results_df["Prediction"].isin(["REAL", "FAKE"])
                        ]

                        if not valid_rows.empty:
                            st.info(
                                "Frame results summarize sampled frames only; "
                                "they do not establish that the entire video "
                                "is authentic or manipulated."
                            )

                        st.download_button(
                            "Download frame results (CSV)",
                            data=results_df.to_csv(index=False).encode("utf-8"),
                            file_name="deeptrace_video_results.csv",
                            mime="text/csv"
                        )

                        try:
                            excel_bytes = None
                            import io

                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(
                                excel_buffer,
                                engine="openpyxl"
                            ) as writer:
                                results_df.to_excel(
                                    writer,
                                    index=False,
                                    sheet_name="Frame Results"
                                )

                            excel_bytes = excel_buffer.getvalue()

                            st.download_button(
                                "Download frame results (Excel)",
                                data=excel_bytes,
                                file_name="deeptrace_video_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                        except Exception:
                            st.caption(
                                "Excel export unavailable. CSV download is ready."
                            )

                if analyze_video_audio:
                    st.subheader("Video audio analysis")
                    audio_path = video_path + ".wav"

                    try:
                        with st.spinner("Extracting audio from video..."):
                            extract_audio_from_video(
                                video_path,
                                audio_path
                            )

                        with st.spinner("Analyzing extracted audio..."):
                            audio_result = predict_audio_file(audio_path)

                        st.write(f"Audio result: **{audio_result['label']}**")
                        st.metric(
                            "Audio fake score",
                            f"{audio_result['fake_score']}%"
                        )
                        st.metric(
                            "Audio real score",
                            f"{audio_result['real_score']}%"
                        )

                    except Exception as audio_error:
                        st.warning(
                            "Video frames were processed independently, "
                            "but audio analysis failed. The video may have "
                            "no audio track, or the audio model may have "
                            "encountered an error."
                        )
                        st.caption(str(audio_error))

                report = {
                    "video_filename": uploaded_video.name,
                    "frame_results": frame_results
                }

                st.download_button(
                    "Download JSON report",
                    data=json.dumps(report, indent=2),
                    file_name="deeptrace_video_report.json",
                    mime="application/json"
                )

            except Exception as error:
                st.error(f"Video analysis failed: {error}")

            finally:
                for path in [video_path, audio_path]:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass


# ---------------- AUDIO TAB ----------------

with audio_tab:
    st.subheader("Analyze an audio recording")

    uploaded_audio = st.file_uploader(
        "Upload WAV, MP3, FLAC, or M4A",
        type=["wav", "mp3", "flac", "m4a"],
        key="audio_upload"
    )

    if uploaded_audio is not None:
        if st.button("Analyze audio", key="analyze_audio"):
            audio_path = None

            try:
                suffix = os.path.splitext(uploaded_audio.name)[1] or ".wav"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_audio:
                    temp_audio.write(uploaded_audio.getbuffer())
                    audio_path = temp_audio.name

                with st.spinner("Analyzing audio..."):
                    result = predict_audio_file(audio_path)

                st.subheader(f"Result: {result['label']}")
                st.metric("Fake score", f"{result['fake_score']}%")
                st.metric("Real score", f"{result['real_score']}%")

            except Exception as error:
                st.error(f"Audio analysis failed: {error}")

            finally:
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except OSError:
                        pass

