
import os
import io
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


# =========================================================
# CONFIG
# =========================================================

IMAGE_MODEL_PATH = "deeptrace_model_scripted.pt"
AUDIO_REPO_ID = "deeptrace-research/deep-trace-audio"
AUDIO_MODEL_FILENAME = "best_model.onnx"

# Verify these indices against the models' training labels.
FAKE_INDEX = 0
REAL_INDEX = 1
AUDIO_FAKE_INDEX = 1

UNCERTAIN_MARGIN = 0.10
MAX_VIDEO_FRAMES = 10
FRAME_INTERVAL = 15
SAMPLE_RATE = 16000
AUDIO_SAMPLES = 48000  # 3 seconds


# =========================================================
# PAGE + NETFLIX FORENSIC THEME
# =========================================================

st.set_page_config(
    page_title="DEEP-TRACE | Forensic Lab",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b0b0d;
    color: #f5f5f5;
}
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 800 !important;
}
p, label, .stMarkdown {
    color: #e5e5e5;
}
[data-testid="stDecoration"] {
    background: #e50914;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 18px;
    border-bottom: 1px solid #333333;
}
.stTabs [data-baseweb="tab"] {
    color: #bbbbbb;
    font-weight: 700;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #e50914 !important;
    border-bottom: 3px solid #e50914 !important;
}
.stButton > button, .stDownloadButton > button {
    background: #e50914 !important;
    color: white !important;
    border: 1px solid #e50914 !important;
    border-radius: 5px !important;
    font-weight: 700 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #b20710 !important;
    color: white !important;
}
[data-testid="stFileUploader"] {
    background: #171719;
    border: 1px solid #383838;
    border-radius: 9px;
    padding: 12px;
}
[data-testid="stMetric"] {
    background: #171719;
    border: 1px solid #383838;
    border-left: 4px solid #e50914;
    border-radius: 8px;
    padding: 15px;
}
[data-testid="stMetricLabel"] {
    color: #cccccc !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #383838;
    border-radius: 8px;
}
.stAlert {
    background: #191316;
    color: #eeeeee;
    border: 1px solid #512027;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background:linear-gradient(110deg,#21080c,#101012 65%,#1b1b20);
    border:1px solid #402026;
    border-left:6px solid #e50914;
    border-radius:10px;
    padding:25px;
    margin-bottom:20px;">
    <div style="color:#e50914;font-size:12px;font-weight:800;
                letter-spacing:4px;">
        DIGITAL FORENSICS • MEDIA INTELLIGENCE
    </div>
    <div style="color:white;font-size:38px;font-weight:900;
                letter-spacing:2px;margin-top:8px;">
        🔎 DEEP-TRACE
    </div>
    <div style="color:#cccccc;font-size:15px;margin-top:8px;">
        AI-ASSISTED DEEPFAKE INVESTIGATION SYSTEM
    </div>
    <div style="color:#888888;font-size:12px;margin-top:14px;
                letter-spacing:1px;">
        IMAGE ANALYSIS / VIDEO ANALYSIS / AUDIO ANALYSIS
    </div>
</div>
<div style="background:#171012;border-left:4px solid #e50914;
padding:10px 14px;border-radius:5px;color:#d8c8c8;font-size:13px;">
FORENSIC NOTICE: AI predictions are estimates, not conclusive proof.
</div>
""", unsafe_allow_html=True)


# =========================================================
# IMAGE MODEL
# =========================================================

@st.cache_resource
def load_image_model():
    if not os.path.exists(IMAGE_MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {IMAGE_MODEL_PATH}. Add it to your repository."
        )

    model = torch.jit.load(IMAGE_MODEL_PATH, map_location="cpu")
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


def prediction_label(real_score, fake_score):
    if abs(real_score - fake_score) < UNCERTAIN_MARGIN:
        return "UNCERTAIN"
    return "FAKE" if fake_score > real_score else "REAL"


def predict_image(image):
    model = load_image_model()
    tensor = image_transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        output = model(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        output = output.detach().cpu()

        if output.numel() == 1:
            fake_score = float(torch.sigmoid(output).item())
            real_score = 1.0 - fake_score
        else:
            probs = F.softmax(output.reshape(1, -1), dim=1)[0]

            if len(probs) <= max(FAKE_INDEX, REAL_INDEX):
                raise ValueError("Unexpected image model output shape.")

            fake_score = float(probs[FAKE_INDEX])
            real_score = float(probs[REAL_INDEX])

    return {
        "label": prediction_label(real_score, fake_score),
        "real_score": round(real_score * 100, 2),
        "fake_score": round(fake_score * 100, 2)
    }


# =========================================================
# AUDIO MODEL
# =========================================================

@st.cache_resource
def load_audio_model():
    token = st.secrets.get("HF_TOKEN", None)

    model_path = hf_hub_download(
        repo_id=AUDIO_REPO_ID,
        filename=AUDIO_MODEL_FILENAME,
        token=token
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1

    return ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )


def predict_audio_file(audio_path):
    session = load_audio_model()

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("No usable audio was found.")

    if len(audio) < AUDIO_SAMPLES:
        audio = np.pad(audio, (0, AUDIO_SAMPLES - len(audio)))
    else:
        audio = audio[:AUDIO_SAMPLES]

    audio = audio.astype(np.float32)
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-5)

    model_input_info = session.get_inputs()[0]
    input_name = model_input_info.name
    input_shape = model_input_info.shape

    if len(input_shape) == 1:
        model_input = audio
    elif len(input_shape) == 2:
        model_input = audio[np.newaxis, :]
    elif len(input_shape) == 3:
        model_input = audio[np.newaxis, np.newaxis, :]
    else:
        raise ValueError(
            f"Unsupported ONNX input shape: {input_shape}"
        )

    outputs = session.run(
        None,
        {input_name: model_input.astype(np.float32)}
    )

    if not outputs:
        raise ValueError("Audio model returned no output.")

    values = np.asarray(outputs[0]).squeeze()

    if values.size == 1:
        value = float(values.reshape(-1)[0])
        fake_score = float(
            1.0 / (1.0 + np.exp(-np.clip(value, -80, 80)))
        )
        real_score = 1.0 - fake_score
    else:
        values = values.reshape(-1).astype(np.float64)
        values -= np.max(values)
        probs = np.exp(values)
        probs /= probs.sum()

        if len(probs) != 2:
            raise ValueError(
                f"Expected 2 audio classes; model returned {len(probs)}. "
                "Check the model's expected output labels."
            )

        fake_score = float(probs[AUDIO_FAKE_INDEX])
        real_score = float(probs[1 - AUDIO_FAKE_INDEX])

    return {
        "label": prediction_label(real_score, fake_score),
        "real_score": round(real_score * 100, 2),
        "fake_score": round(fake_score * 100, 2)
    }


# =========================================================
# SHARED RESULT DISPLAY
# =========================================================

def show_prediction(result):
    label = result["label"]

    if label == "FAKE":
        st.error("🔴 Prediction: FAKE")
    elif label == "REAL":
        st.success("🟢 Prediction: REAL")
    else:
        st.warning("🟡 Prediction: UNCERTAIN")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Fake score", f"{result['fake_score']:.2f}%")

    with col2:
        st.metric("Real score", f"{result['real_score']:.2f}%")


# =========================================================
# VIDEO HELPERS
# =========================================================

def extract_audio_from_video(video_path, audio_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg, "-y",
        "-i", video_path,
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
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
        raise RuntimeError(
            "No readable audio track found. This video may have no sound."
        )

    if os.path.getsize(audio_path) == 0:
        raise RuntimeError("The extracted audio track is empty.")


def analyze_video_frames(video_path, progress):
    reader = imageio.get_reader(video_path)
    results = []

    try:
        try:
            total_frames = reader.count_frames()
            indices = list(
                range(0, total_frames, FRAME_INTERVAL)
            )[:MAX_VIDEO_FRAMES]
        except Exception:
            indices = list(range(MAX_VIDEO_FRAMES))

        for position, frame_number in enumerate(indices):
            try:
                frame = reader.get_data(frame_number)
                frame_image = Image.fromarray(
                    np.asarray(frame).astype(np.uint8)
                ).convert("RGB")

                result = predict_image(frame_image)

                results.append({
                    "Frame": frame_number,
                    "Prediction": result["label"],
                    "Real score (%)": result["real_score"],
                    "Fake score (%)": result["fake_score"]
                })

            except Exception as error:
                results.append({
                    "Frame": frame_number,
                    "Prediction": "ERROR",
                    "Real score (%)": None,
                    "Fake score (%)": None,
                    "Error": str(error)
                })

            progress.progress(
                min((position + 1) / max(len(indices), 1), 1.0)
            )

    finally:
        reader.close()

    return pd.DataFrame(results)


def show_frame_downloads(df):
    st.download_button(
        "⬇ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="deeptrace_video_frames.csv",
        mime="text/csv"
    )

    try:
        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Frame Results")

        st.download_button(
            "⬇ Download Excel",
            data=buffer.getvalue(),
            file_name="deeptrace_video_frames.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception:
        st.caption("Excel export unavailable. CSV is available.")


# =========================================================
# TABS
# =========================================================

image_tab, video_tab, audio_tab = st.tabs([
    "🖼️ IMAGE FORENSICS",
    "🎬 VIDEO FORENSICS",
    "🎙️ AUDIO FORENSICS"
])


# =========================================================
# IMAGE TAB
# =========================================================

with image_tab:
    st.subheader("Image authenticity analysis")

    uploaded_image = st.file_uploader(
        "Upload JPG, PNG, WEBP, or BMP",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="image_upload"
    )

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Evidence image", width=500)

            if st.button("🔎 Analyze image", key="analyze_image"):
                with st.spinner("Analyzing image..."):
                    result = predict_image(image)
                show_prediction(result)

        except Exception as error:
            st.error(f"Image analysis failed: {error}")


# =========================================================
# VIDEO TAB
# =========================================================

with video_tab:
    st.subheader("Video authenticity analysis")

    uploaded_video = st.file_uploader(
        "Upload MP4, MOV, AVI, MKV, or WEBM",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="video_upload"
    )

    analyze_frames = st.checkbox(
        "Analyze sampled video frames",
        value=True
    )

    analyze_video_audio = st.checkbox(
        "Also analyze audio if the video contains sound",
        value=False
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("🔎 Analyze video", key="analyze_video"):
            video_path = None
            audio_path = None

            try:
                suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_video:
                    temp_video.write(uploaded_video.getbuffer())
                    video_path = temp_video.name

                if analyze_frames:
                    st.write("**Frame-by-frame analysis**")
                    progress = st.progress(0)

                    with st.spinner("Analyzing sampled frames..."):
                        df = analyze_video_frames(video_path, progress)

                    progress.empty()

                    if df.empty:
                        st.warning("No video frames could be analyzed.")
                    else:
                        st.dataframe(df, use_container_width=True)

                        valid = df[df["Prediction"].isin(["REAL", "FAKE"])]

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                "Frames labelled fake",
                                int((valid["Prediction"] == "FAKE").sum())
                            )

                        with col2:
                            st.metric(
                                "Frames labelled real",
                                int((valid["Prediction"] == "REAL").sum())
                            )

                        show_frame_downloads(df)

                if analyze_video_audio:
                    st.subheader("🎙️ Audio track analysis")
                    audio_path = video_path + ".wav"

                    try:
                        with st.spinner("Extracting audio from video..."):
                            extract_audio_from_video(video_path, audio_path)

                        st.success("Audio extracted. Running detection...")

                        with st.spinner("Analyzing audio..."):
                            audio_result = predict_audio_file(audio_path)

                        show_prediction(audio_result)

                    except Exception as audio_error:
                        st.error("Audio analysis could not be completed.")
                        st.code(str(audio_error))
                        st.info(
                            "The video may not contain audio. Frame analysis "
                            "can still work. Try a separate WAV or MP3 in "
                            "the Audio Forensics tab."
                        )

                st.download_button(
                    "⬇ Download JSON report",
                    data=json.dumps({
                        "video_filename": uploaded_video.name,
                        "note": "Sampled frame predictions are not proof of authenticity."
                    }, indent=2),
                    file_name="deeptrace_report.json",
                    mime="application/json"
                )

            except Exception as error:
                st.error(f"Video analysis failed: {error}")

            finally:
                for path in (video_path, audio_path):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass


# =========================================================
# AUDIO TAB
# =========================================================

with audio_tab:
    st.subheader("Audio authenticity analysis")

    uploaded_audio = st.file_uploader(
        "Upload WAV, MP3, FLAC, M4A, or OGG",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        key="audio_upload"
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if st.button("🔎 Analyze audio", key="analyze_audio"):
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

                show_prediction(result)

                st.caption(
                    "The model analyzes a three-second audio segment. "
                    "Confirm its class mapping before relying on predictions."
                )

            except Exception as error:
                st.error(f"Audio analysis failed: {error}")
                st.code(str(error))

            finally:
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except OSError:
                        pass


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<hr>
<div style="text-align:center;color:#777;font-size:12px;padding:12px;">
DEEP-TRACE • DIGITAL FORENSICS LAB • AI-ASSISTED MEDIA ANALYSIS
</div>
""", unsafe_allow_html=True)
