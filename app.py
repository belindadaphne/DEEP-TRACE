import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import tempfile
import os
import subprocess
import glob

# ============================================================
# DEEP-TRACE
# Deepfake Image + Video Detection
# ============================================================

st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🔍",
    layout="wide"
)

MODEL_PATH = "deeptrace_model_scripted.pt"

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model


model = load_model()

# ------------------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Your training output showed:
# CLASS ORDER = ['fake', 'real']
CLASS_NAMES = ["fake", "real"]


def predict_image(image):
    image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    # Handle tensor output
    if isinstance(output, (tuple, list)):
        output = output[0]

    output = output.float()

    # Convert model output to probabilities
    probabilities = torch.softmax(output, dim=1)[0]

    fake_probability = float(probabilities[0])
    real_probability = float(probabilities[1])

    if fake_probability >= real_probability:
        label = "FAKE"
        confidence = fake_probability
    else:
        label = "REAL"
        confidence = real_probability

    return label, confidence, fake_probability, real_probability


# ------------------------------------------------------------
# VIDEO FRAME EXTRACTION
# ------------------------------------------------------------

def extract_video_frames(video_path, max_frames=12):

    frame_dir = tempfile.mkdtemp()

    output_pattern = os.path.join(
        frame_dir,
        "frame_%03d.jpg"
    )

    # Use ffmpeg instead of cv2
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        "fps=1",
        "-frames:v",
        str(max_frames),
        output_pattern,
        "-y"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    frames = sorted(
        glob.glob(
            os.path.join(frame_dir, "*.jpg")
        )
    )

    return frames


def predict_video(video_path):

    frame_paths = extract_video_frames(
        video_path,
        max_frames=12
    )

    if not frame_paths:
        return None

    fake_scores = []
    real_scores = []

    for frame_path in frame_paths:

        try:
            image = Image.open(frame_path)

            label, confidence, fake_prob, real_prob = \
                predict_image(image)

            fake_scores.append(fake_prob)
            real_scores.append(real_prob)

        except Exception:
            continue

    if not fake_scores:
        return None

    avg_fake = sum(fake_scores) / len(fake_scores)
    avg_real = sum(real_scores) / len(real_scores)

    if avg_fake >= avg_real:
        final_label = "FAKE"
        final_confidence = avg_fake
    else:
        final_label = "REAL"
        final_confidence = avg_real

    return (
        final_label,
        final_confidence,
        avg_fake,
        avg_real,
        len(fake_scores)
    )


# ============================================================
# UI
# ============================================================

st.title("🔍 DEEP-TRACE")

st.subheader("AI-Powered Deepfake Detection System")

st.write(
    "Upload an image or video and DEEP-TRACE will "
    "analyze it using the trained deepfake detection model."
)

st.divider()

# ------------------------------------------------------------
# MODEL STATUS
# ------------------------------------------------------------

if model is None:

    st.error(
        "❌ Trained model not found."
    )

    st.info(
        "Make sure deeptrace_model_scripted.pt "
        "is uploaded beside app.py in GitHub."
    )

    st.stop()

else:

    st.success(
        "✅ DEEP-TRACE trained model loaded successfully."
    )


# ------------------------------------------------------------
# UPLOAD TYPE
# ------------------------------------------------------------

option = st.radio(
    "Select input type:",
    ["Image", "Video"],
    horizontal=True
)

st.divider()


# ============================================================
# IMAGE
# ============================================================

if option == "Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "🔍 Analyze Image",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                try:

                    label, confidence, fake_prob, real_prob = \
                        predict_image(image)

                    st.divider()

                    st.subheader(
                        "DEEP-TRACE RESULT"
                    )

                    if label == "FAKE":

                        st.error(
                            f"🚨 RESULT: {label}"
                        )

                    else:

                        st.success(
                            f"✅ RESULT: {label}"
                        )

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Fake Probability",
                            f"{fake_prob * 100:.2f}%"
                        )

                    with col2:

                        st.metric(
                            "Real Probability",
                            f"{real_prob * 100:.2f}%"
                        )

                    st.progress(
                        min(max(float(confidence), 0.0), 1.0)
                    )

                except Exception as e:

                    st.error(
                        "Prediction failed."
                    )

                    st.code(str(e))


# ============================================================
# VIDEO
# ============================================================

else:

    uploaded_file = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv"
        ]
    )

    if uploaded_file is not None:

        st.video(uploaded_file)

        if st.button(
            "🎬 Analyze Video",
            use_container_width=True
        ):

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.read()
                    )

                    temp_path = temp_file.name

                with st.spinner(
                    "Extracting video frames and analyzing..."
                ):

                    result = predict_video(
                        temp_path
                    )

                if result is None:

                    st.error(
                        "Could not extract frames from this video."
                    )

                else:

                    (
                        label,
                        confidence,
                        fake_prob,
                        real_prob,
                        frame_count
                    ) = result

                    st.divider()

                    st.subheader(
                        "DEEP-TRACE VIDEO RESULT"
                    )

                    if label == "FAKE":

                        st.error(
                            f"🚨 RESULT: {label}"
                        )

                    else:

                        st.success(
                            f"✅ RESULT: {label}"
                        )

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Fake Probability",
                            f"{fake_prob * 100:.2f}%"
                        )

                    with col2:

                        st.metric(
                            "Real Probability",
                            f"{real_prob * 100:.2f}%"
                        )

                    with col3:

                        st.metric(
                            "Frames Analyzed",
                            frame_count
                        )

                    st.progress(
                        min(max(float(confidence), 0.0), 1.0)
                    )

            except Exception as e:

                st.error(
                    "Video analysis failed."
                )

                st.code(str(e))

            finally:

                if temp_path and os.path.exists(temp_path):

                    os.remove(temp_path)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DEEP-TRACE | Deepfake Detection using a trained "
    "FaceForensics++ based model"
)
