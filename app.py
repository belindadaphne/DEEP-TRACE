
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


# =========================================================
# CONFIGURATION
# =========================================================

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

# Keep the existing audio output assumption from your original code.
# Verify the audio class order against the model documentation.
AUDIO_FAKE_INDEX = 1

UNCERTAIN_MARGIN = 0.10
MAX_VIDEO_FRAMES = 10
FRAME_INTERVAL = 15

AUDIO_SAMPLE_RATE = 16000
AUDIO_MAX_LENGTH = 48000

st.title("🔎 DEEP-TRACE")
st.caption("AI-assisted deepfake screening")


# =========================================================
# LOAD IMAGE / VIDEO MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    loaded_model = torch.jit.load(
        MODEL_PATH,
        map_location="cpu"
    )
    loaded_model.eval()
    return loaded_model


try:
    model = load_model()
    st.success("Image/video detection model loaded")
except Exception as e:
    st.error(f"Could not load image/video model: {e}")
    st.stop()


# Keep your original image preprocessing unchanged.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# IMAGE PREDICTION
# =========================================================

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


# =========================================================
# AUDIO MODEL — LOAD ONLY WHEN NEEDED
# =========================================================

@st.cache_resource
def load_audio_model():
    token = st.secrets.get("HF_TOKEN")

    # The repository is public now, so a token is optional.
    # If one is configured, use it.
    model_file = hf_hub_download(
        repo_id=AUDIO_REPO_ID,
        filename=AUDIO_MODEL_FILENAME,
        token=token if token else None
    )

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1

    return ort.InferenceSession(
        model_file,
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )


def predict_audio_file(audio_path):
    audio, _ = librosa.load(
        audio_path,
        sr=AUDIO_SAMPLE_RATE,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("No audio samples were found.")

    duration = len(audio) / AUDIO_SAMPLE_RATE

    # This model input uses only the first 3 seconds.
    if len(audio) > AUDIO_MAX_LENGTH:
        audio = audio[:AUDIO_MAX_LENGTH]
    elif len(audio) < AUDIO_MAX_LENGTH:
        audio = np.pad(
            audio,
            (0, AUDIO_MAX_LENGTH - len(audio))
        )

    audio = (
        audio - np.mean(audio)
    ) / np.sqrt(np.var(audio) + 1e-5)

    audio = audio.astype(np.float32).reshape(
        1, AUDIO_MAX_LENGTH
    )

    session = load_audio_model()
    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {input_name: audio}
    )

    logits = np.asarray(outputs[0]).reshape(1, -1)

    if logits.shape[1] < 2:
        raise ValueError(
            f"Expected two audio classes; got output shape {logits.shape}"
        )

    shifted = logits - np.max(
        logits, axis=1, keepdims=True
    )
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    if AUDIO_FAKE_INDEX >= probs.shape[1]:
        raise ValueError(
            "The configured fake class index does not match model output."
        )

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
        "analyzed_seconds": min(round(duration, 2), 3.0),
        "sample_rate": AUDIO_SAMPLE_RATE
    }


# =========================================================
# AUDIO EXTRACTION — HANDLE VIDEOS WITHOUT AUDIO
# =========================================================

def extract_audio_from_video(video_path, audio_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", video_path,
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        audio_path
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if completed.returncode != 0:
        details = (
            completed.stderr[-1000:]
            if completed.stderr
            else "No details available."
        )

        if (
            "matches no streams" in details.lower()
            or "does not contain any stream" in details.lower()
            or "stream map" in details.lower()
        ):
            raise ValueError(
                "This video does not appear to contain an audio track."
            )

        raise RuntimeError(
            "FFmpeg could not extract the audio track. "
            + details
        )

    if (
        not os.path.exists(audio_path)
        or os.path.getsize(audio_path) == 0
    ):
        raise ValueError(
            "No usable audio track was extracted from this video."
        )


# =========================================================
# DOWNLOAD HELPERS
# =========================================================

def csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def excel_bytes(dataframe):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:
        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Frame Results"
        )

    output.seek(0)
    return output.getvalue()


# =========================================================
# APP MODE
# =========================================================

mode = st.radio(
    "Choose analysis type",
    [
        "Image detection",
        "Video detection",
        "Audio detection"
    ],
    horizontal=True
)


# =========================================================
# IMAGE DETECTION
# =========================================================

if mode == "Image detection":

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded is not None:
        data = uploaded.getvalue()

        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            st.image(
                image,
                caption="Uploaded image",
                width="stretch"
            )
        except Exception as e:
            st.error(f"Could not open this image: {e}")
            st.stop()

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


# =========================================================
# VIDEO DETECTION
# =========================================================

elif mode == "Video detection":

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded is not None:
        video_data = uploaded.getvalue()

        st.video(video_data)

        analyze_visual = st.checkbox(
            "Analyze video frames",
            value=True
        )

        analyze_audio = st.checkbox(
            "Extract and analyze video's audio track",
            value=False
        )

        st.caption(
            "Tip: Keep audio analysis unchecked for videos without sound. "
            "You can analyze the video frames independently."
        )

        if st.button("Analyze video", type="primary"):

            video_path = None
            audio_path = None
            reader = None
            results_df = None

            report = {
                "file_name": uploaded.name,
                "file_sha256": file_hash(video_data),
                "analysis_time_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "visual_analysis": None,
                "audio_analysis": None,
                "audio_error": None
            }

            try:
                suffix = (
                    os.path.splitext(uploaded.name)[1]
                    or ".mp4"
                )

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False
                ) as temp_file:
                    temp_file.write(video_data)
                    video_path = temp_file.name

                # -----------------------------------------
                # FRAME ANALYSIS
                # -----------------------------------------

                if analyze_visual:
                    rows = []

                    with st.spinner("Analyzing video frames..."):
                        reader = imageio.get_reader(
                            video_path,
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
                                "Timestamp (seconds)": round(
                                    timestamp, 2
                                ),
                                "Prediction": result["verdict"],
                                "REAL score (%)": round(
                                    result["real_score"] * 100, 2
                                ),
                                "FAKE score (%)": round(
                                    result["fake_score"] * 100, 2
                                )
                            })

                    if rows:
                        results_df = pd.DataFrame(rows)

                        real_average = (
                            results_df["REAL score (%)"].mean() / 100
                        )
                        fake_average = (
                            results_df["FAKE score (%)"].mean() / 100
                        )

                        if abs(
                            real_average - fake_average
                        ) < UNCERTAIN_MARGIN:
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

                        st.dataframe(
                            results_df,
                            width="stretch"
                        )

                        # CSV download
                        st.download_button(
                            "Download frame results (CSV)",
                            data=csv_bytes(results_df),
                            file_name="deeptrace_frame_results.csv",
                            mime="text/csv"
                        )

                        # Excel download
                        try:
                            xlsx_data = excel_bytes(results_df)

                            st.download_button(
                                "Download frame results (Excel)",
                                data=xlsx_data,
                                file_name="deeptrace_frame_results.xlsx",
                                mime=(
                                    "application/vnd.openxmlformats-officedocument."
                                    "spreadsheetml.sheet"
                                )
                            )

                        except Exception as e:
                            st.warning(
                                "Excel export is unavailable. "
                                "Use the CSV download instead. "
                                f"Details: {e}"
                            )

                        report["visual_analysis"] = {
                            "verdict": verdict,
                            "average_real_score": real_average,
                            "average_fake_score": fake_average,
                            "frame_results": rows
                        }

                    else:
                        st.warning(
                            "No video frames could be analyzed."
                        )

                # -----------------------------------------
                # AUDIO ANALYSIS
                # Audio failure must not erase frame results.
                # -----------------------------------------

                if analyze_audio:
                    st.subheader("Audio track analysis")

                    try:
                        with st.spinner(
                            "Extracting audio and analyzing it..."
                        ):
                            fd, audio_path = tempfile.mkstemp(
                                suffix=".wav"
                            )
                            os.close(fd)
                            os.remove(audio_path)

                            extract_audio_from_video(
                                video_path,
                                audio_path
                            )

                            audio_result = predict_audio_file(
                                audio_path
                            )

                        st.success("Audio track extracted.")
                        show_result(audio_result)

                        st.write(
                            "Audio duration: "
                            f"{audio_result['duration_seconds']} seconds"
                        )
                        st.write(
                            "Audio analyzed by model: "
                            f"{audio_result['analyzed_seconds']} seconds"
                        )

                        with open(audio_path, "rb") as audio_file:
                            st.audio(
                                audio_file.read(),
                                format="audio/wav"
                            )

                        report["audio_analysis"] = audio_result

                    except Exception as audio_error:
                        report["audio_error"] = str(audio_error)

                        st.warning(
                            "Audio analysis could not be completed. "
                            "Video frame results are still available."
                        )
                        st.error(str(audio_error))

                # -----------------------------------------
                # COMBINED REPORT
                # -----------------------------------------

                if (
                    report["visual_analysis"] is not None
                    or report["audio_analysis"] is not None
                    or report["audio_error"] is not None
                ):
                    st.download_button(
                        "Download combined forensic report (JSON)",
                        json.dumps(report, indent=2),
                        file_name="deeptrace_video_forensic_report.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"Video analysis failed: {e}")

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


# =========================================================
# STANDALONE AUDIO DETECTION
# =========================================================

else:

    st.subheader("🎙️ Audio deepfake detection")

    st.write(
        "Upload a WAV, FLAC, or MP3 file. "
        "For audio inside a video, use Video detection."
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
                suffix = (
                    os.path.splitext(uploaded.name)[1]
                    or ".wav"
                )

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False
                ) as temp_file:
                    temp_file.write(audio_bytes)
                    audio_path = temp_file.name

                with st.spinner(
                    "Loading audio model and analyzing audio..."
                ):
                    result = predict_audio_file(audio_path)

                show_result(result)

                st.write(
                    "Audio analyzed by model: "
                    f"{result['analyzed_seconds']} seconds"
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
                    "Check that best_model.onnx exists in the configured "
                    "Hugging Face repository and that the model can be "
                    "downloaded. If the app runs out of memory, avoid "
                    "repeatedly loading the audio model."
                )

            finally:
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DEEP-TRACE is an AI screening tool. Predictions can be wrong "
    "and should not be treated as definitive proof of authenticity."
)
