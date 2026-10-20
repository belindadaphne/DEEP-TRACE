
import io
import os
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

st.set_page_config(page_title="DEEP-TRACE", page_icon="🔎", layout="wide")
st.title("🔎 DEEP-TRACE")
st.caption("Experimental deepfake screening for images and videos")

MODEL_PATH = "deeptrace_model_scripted.pt"
FAKE_INDEX = 0  # Verify against the model's training code before trusting labels.
REAL_INDEX = 1
UNCERTAIN_MARGIN = 0.10

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    loaded = torch.jit.load(MODEL_PATH, map_location="cpu")
    loaded.eval()
    return loaded

try:
    model = load_model()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image):
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        output = model(tensor)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError("Unsupported model output type")
    values = output.detach().float().cpu()
    if values.ndim == 2:
        values = values[0]
    elif values.ndim != 1:
        raise ValueError(f"Unexpected model output shape: {tuple(values.shape)}")
    raw = values.numpy().astype(float)
    if len(raw) != 2:
        raise ValueError(f"Expected 2 class outputs, got {len(raw)}")
    if np.all(raw >= 0) and np.all(raw <= 1) and np.isclose(raw.sum(), 1.0, atol=1e-3):
        scores = raw
    else:
        shifted = raw - np.max(raw)
        exp_values = np.exp(shifted)
        scores = exp_values / exp_values.sum()
    fake_score = float(scores[FAKE_INDEX])
    real_score = float(scores[REAL_INDEX])
    if abs(fake_score - real_score) < UNCERTAIN_MARGIN:
        verdict = "UNCERTAIN"
    else:
        verdict = "FAKE" if fake_score > real_score else "REAL"
    return {"verdict": verdict, "fake_score": fake_score, "real_score": real_score,
            "raw_output": raw.tolist(), "model_scores": scores.tolist()}

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def report_download(report, filename, label):
    st.download_button(label, data=json.dumps(report, indent=2), file_name=filename, mime="application/json")

# IMAGE
st.header("1. Image Detection")
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key="img")
if image_file:
    image_bytes = image_file.getvalue()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(image, caption="Uploaded image", width=360)
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                result = predict_image(image)
            st.subheader(f"Verdict: {result['verdict']}")
            c1, c2 = st.columns(2)
            c1.metric("REAL score", f"{result['real_score'] * 100:.2f}%")
            c2.metric("FAKE score", f"{result['fake_score'] * 100:.2f}%")
            with st.expander("Diagnostic model output"):
                st.write("Raw output:", result["raw_output"])
                st.write("Scores:", result["model_scores"])
                st.caption("Class order is assumed, not verified. These scores do not prove authenticity.")
            report_download({"file_name": image_file.name, "type": "image", "sha256": sha256(image_bytes),
                             "timestamp_utc": datetime.now(timezone.utc).isoformat(), **result,
                             "warning": "Experimental result; not proof of authenticity."},
                            "deep_trace_image_report.json", "Download image report (JSON)")
    except Exception as exc:
        st.error(f"Image analysis failed: {exc}")

# VIDEO
st.divider()
st.header("2. Video Detection")
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv", "webm"], key="vid")
if video_file:
    video_bytes = video_file.getvalue()
    st.video(video_bytes)
    if st.button("Analyze Video", type="primary"):
        temp_path = None
        reader = None
        try:
            suffix = os.path.splitext(video_file.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(video_bytes)
                temp_path = tmp.name
            reader = imageio.get_reader(temp_path, format="ffmpeg")
            meta = reader.get_meta_data()
            fps = float(meta.get("fps") or 0)
            sample_every = max(1, int(round(fps * 0.5))) if fps > 0 else 15
            rows = []
            for idx, frame in enumerate(reader):
                if idx % sample_every != 0:
                    continue
                prediction = predict_image(Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB"))
                rows.append({"Frame": idx, "Timestamp (seconds)": round(idx / fps, 2) if fps > 0 else None,
                             "Prediction": prediction["verdict"], "REAL score (%)": round(prediction["real_score"] * 100, 2),
                             "FAKE score (%)": round(prediction["fake_score"] * 100, 2),
                             "Score difference (%)": round(abs(prediction["real_score"] - prediction["fake_score"]) * 100, 2),
                             "Raw output": json.dumps(prediction["raw_output"])})
                if len(rows) >= 10:
                    break
            if not rows:
                raise ValueError("No frames could be decoded from this video")
            df = pd.DataFrame(rows)
            real_avg = float(df["REAL score (%)"].mean() / 100)
            fake_avg = float(df["FAKE score (%)"].mean() / 100)
            if abs(real_avg - fake_avg) < UNCERTAIN_MARGIN:
                verdict = "UNCERTAIN"
            else:
                verdict = "FAKE" if fake_avg > real_avg else "REAL"
            st.subheader(f"Overall video verdict: {verdict}")
            c1, c2 = st.columns(2)
            c1.metric("Average REAL score", f"{real_avg * 100:.2f}%")
            c2.metric("Average FAKE score", f"{fake_avg * 100:.2f}%")
            st.write(f"Frames analyzed: {len(rows)} · FPS: {fps:.3f}" if fps > 0 else f"Frames analyzed: {len(rows)} · FPS unavailable")
            st.dataframe(df.drop(columns=["Raw output"]), use_container_width=True, hide_index=True)
            with st.expander("Diagnostic model output"):
                st.write("First sampled frame raw output:", rows[0]["Raw output"])
                st.write(f"Assumed order: index {FAKE_INDEX}=FAKE, index {REAL_INDEX}=REAL")
            st.subheader("File integrity (SHA-256)")
            st.code(sha256(video_bytes))
            report = {"file_name": video_file.name, "type": "video", "sha256": sha256(video_bytes),
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(), "fps": fps,
                      "frames_analyzed": len(rows), "overall_verdict": verdict,
                      "average_real_score": real_avg, "average_fake_score": fake_avg,
                      "frame_results": rows, "warning": "Experimental result; not proof of authenticity."}
            report_download(report, "deep_trace_video_report.json", "Download forensic report (JSON)")
            st.download_button("Download frame results (CSV)", data=df.to_csv(index=False),
                               file_name="deep_trace_frame_results.csv", mime="text/csv")
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

# AUDIO
st.divider()
st.header("3. Audio Deepfake Detection")
st.warning("Audio upload is available, but a trained audio deepfake model is not configured. This section does not label audio as real or fake.")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"], key="aud")
if audio_file:
    audio_bytes = audio_file.getvalue()
    st.audio(audio_bytes)
    st.write(f"File: {audio_file.name}")
    st.write(f"Size: {len(audio_bytes):,} bytes")
    st.write("SHA-256:")
    st.code(sha256(audio_bytes))
    st.info("To perform actual audio deepfake detection, add and load a trained audio classification model. File metadata alone cannot determine whether a voice is synthetic.")
    audio_report = {"file_name": audio_file.name, "type": "audio", "sha256": sha256(audio_bytes),
                    "size_bytes": len(audio_bytes), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "analysis_status": "metadata_only_no_audio_detection_model",
                    "warning": "No real/fake audio prediction was performed."}
    report_download(audio_report, "deep_trace_audio_report.json", "Download audio file report (JSON)")

st.divider()
st.caption("DEEP-TRACE is an experimental screening tool. Predictions can be wrong and are not proof of authenticity.")
