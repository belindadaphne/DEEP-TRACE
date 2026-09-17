import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import cv2
import tempfile
import os
import hashlib

# ============================================================
# DEEP-TRACE
# AI BASED IMAGE & VIDEO DEEPFAKE DETECTION
# ============================================================

st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "deeptrace_model_scripted.pt"

# IMPORTANT:
# Your training output showed:
# CLASS ORDER: ['fake', 'real']
CLASS_NAMES = ["fake", "real"]

IMAGE_SIZE = 224

# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .fake-box {
        background-color: #ffe5e5;
        border: 2px solid #ff4b4b;
    }

    .real-box {
        background-color: #e5ffe8;
        border: 2px solid #21c55d;
    }

    .metric {
        font-size: 25px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ DEEP-TRACE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Image & Video Deepfake Detection System'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        model = torch.jit.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )

        model.eval()

        return model

    except Exception as e:
        st.error("Model loading failed.")
        st.code(str(e))
        return None


model = load_model()

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# FILE HASH
# ============================================================

def get_file_hash(file_bytes):

    return hashlib.sha256(file_bytes).hexdigest()


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_image(image):

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)

        # Some TorchScript models return tuples/lists
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Convert output to tensor
        output = torch.as_tensor(output)

        # Make sure shape is [1, number_of_classes]
        if output.dim() == 1:
            output = output.unsqueeze(0)

        probabilities = F.softmax(output, dim=1)

        fake_probability = probabilities[0][0].item()
        real_probability = probabilities[0][1].item()

    if fake_probability > real_probability:

        verdict = "DEEPFAKE"

        confidence = fake_probability

    else:

        verdict = "REAL"

        confidence = real_probability

    return (
        verdict,
        confidence,
        fake_probability,
        real_probability
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    verdict,
    confidence,
    fake_probability,
    real_probability
):

    st.markdown("---")

    if verdict == "DEEPFAKE":

        st.markdown(
            """
            <div class="result-box fake-box">
                <h1>⚠️ DEEPFAKE DETECTED</h1>
                <p class="metric">
                    Model prediction: FAKE
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-box real-box">
                <h1>✅ LIKELY REAL</h1>
                <p class="metric">
                    Model prediction: REAL
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Fake Probability",
            f"{fake_probability * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Real Probability",
            f"{real_probability * 100:.2f}%"
        )


# ============================================================
# IMAGE DETECTION
# ============================================================

def process_image(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # File information
    st.subheader("📄 File Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Filename:**", uploaded_file.name)

        st.write(
            "**Format:**",
            image.format if image.format else "Unknown"
        )

    with col2:

        st.write(
            "**Resolution:**",
            f"{image.width} × {image.height}"
        )

        st.write(
            "**SHA-256:**",
            get_file_hash(file_bytes)[:32] + "..."
        )

    # Prediction
    with st.spinner("🔍 Analyzing image..."):

        try:

            result = predict_image(image)

            display_result(*result)

        except Exception as e:

            st.error("Prediction failed.")

            st.code(str(e))


# ============================================================
# VIDEO DETECTION
# ============================================================

def process_video(uploaded_file):

    video_bytes = uploaded_file.getvalue()

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(video_bytes)

        video_path = temp_file.name

    try:

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            st.error("Could not open the video.")

            return

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25

        duration = total_frames / fps

        st.subheader("🎬 Video Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Frames",
                total_frames
            )

        with col2:

            st.metric(
                "FPS",
                f"{fps:.2f}"
            )

        with col3:

            st.metric(
                "Duration",
                f"{duration:.2f} sec"
            )

        # ====================================================
        # SAMPLE VIDEO FRAMES
        # ====================================================

        max_frames = 12

        if total_frames < max_frames:

            sample_count = total_frames

        else:

            sample_count = max_frames

        frame_positions = torch.linspace(
            0,
            max(total_frames - 1, 0),
            sample_count
        ).long().tolist()

        fake_scores = []
        real_scores = []

        processed = 0

        progress = st.progress(0)

        for frame_number in frame_positions:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            success, frame = cap.read()

            if not success:
                continue

            # OpenCV BGR -> RGB
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(
                frame_rgb
            )

            try:

                result = predict_image(
                    pil_image
                )

                verdict, confidence, fake, real = result

                fake_scores.append(fake)
                real_scores.append(real)

                processed += 1

            except Exception:
                continue

            progress.progress(
                processed / sample_count
            )

        progress.empty()

        cap.release()

        # ====================================================
        # VIDEO RESULT
        # ====================================================

        if processed == 0:

            st.error(
                "No readable frames were found."
            )

            return

        average_fake = sum(fake_scores) / len(
            fake_scores
        )

        average_real = sum(real_scores) / len(
            real_scores
        )

        if average_fake > average_real:

            verdict = "DEEPFAKE"

            confidence = average_fake

        else:

            verdict = "REAL"

            confidence = average_real

        st.subheader("🎯 Video Detection Result")

        display_result(
            verdict,
            confidence,
            average_fake,
            average_real
        )

        # ====================================================
        # FRAME ANALYSIS
        # ====================================================

        st.subheader("📊 Frame Analysis")

        st.write(
            f"Analyzed **{processed}** sampled frames."
        )

        frame_fake_percent = [
            score * 100
            for score in fake_scores
        ]

        st.line_chart(
            frame_fake_percent
        )

        st.caption(
            "The chart shows the model's fake probability "
            "for sampled video frames."
        )

    finally:

        if os.path.exists(video_path):

            os.remove(video_path)


# ============================================================
# MAIN INTERFACE
# ============================================================

if model is None:

    st.error(
        "❌ Trained model file not found."
    )

    st.info(
        "Make sure deeptrace_model_scripted.pt "
        "is in the same GitHub folder as app.py."
    )

    st.stop()


st.success(
    "✅ DEEP-TRACE trained model loaded successfully."
)

# ============================================================
# SELECT INPUT TYPE
# ============================================================

input_type = st.radio(
    "Select Detection Type",
    [
        "🖼️ Image",
        "🎬 Video"
    ],
    horizontal=True
)

# ============================================================
# IMAGE UPLOADER
# ============================================================

if input_type == "🖼️ Image":

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

        process_image(
            uploaded_file
        )

# ============================================================
# VIDEO UPLOADER
# ============================================================

else:

    uploaded_file = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )

    if uploaded_file is not None:

        process_video(
            uploaded_file
        )


# ============================================================
# ABOUT SECTION
# ============================================================

st.markdown("---")

with st.expander("ℹ️ About DEEP-TRACE"):

    st.write(
        """
        DEEP-TRACE is an AI-based deepfake detection
        system designed to analyze digital media.

        The current prototype uses a trained image
        classification model to estimate whether
        an uploaded image is REAL or FAKE.

        For video analysis, multiple frames are sampled
        from the uploaded video and individually analyzed.
        The frame-level predictions are then aggregated
        to produce an overall video prediction.

        Dataset:
        FaceForensics++ C23

        Classes:
        • Fake
        • Real
        """
    )

st.caption(
    "DEEP-TRACE | College Project Prototype"
)
