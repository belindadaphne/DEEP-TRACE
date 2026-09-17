import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import tempfile
import os
import subprocess
import glob
import hashlib

# ============================================================
# DEEP-TRACE
# Netflix Style Deepfake Detection System
# ============================================================

st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM NETFLIX-STYLE CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at 80% 10%,
                rgba(180, 0, 0, 0.18),
                transparent 35%
            ),
            linear-gradient(
                180deg,
                #050505 0%,
                #0b0b0b 45%,
                #111111 100%
            );
        color: #ffffff;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Header */
    .netflix-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0 30px 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 30px;
    }

    .brand {
        font-size: 42px;
        font-weight: 900;
        letter-spacing: 4px;
        color: #e50914;
        text-shadow: 0 0 20px rgba(229,9,20,0.35);
    }

    .tagline {
        color: #b3b3b3;
        font-size: 15px;
        margin-top: 5px;
    }

    .nav-badge {
        background: #e50914;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 1px;
    }

    /* Hero section */
    .hero {
        padding: 55px 45px;
        border-radius: 12px;
        margin-bottom: 30px;

        background:
            linear-gradient(
                90deg,
                rgba(0,0,0,0.98) 0%,
                rgba(0,0,0,0.88) 45%,
                rgba(0,0,0,0.40) 100%
            ),
            radial-gradient(
                circle at 85% 50%,
                rgba(229,9,20,0.35),
                transparent 45%
            );

        border: 1px solid rgba(229,9,20,0.20);
        box-shadow: 0 15px 50px rgba(0,0,0,0.45);
    }

    .hero-title {
        font-size: 48px;
        font-weight: 900;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }

    .hero-title span {
        color: #e50914;
    }

    .hero-text {
        color: #d2d2d2;
        font-size: 18px;
        max-width: 700px;
        line-height: 1.6;
    }

    /* Cards */
    .feature-card {
        background: linear-gradient(
            145deg,
            #181818,
            #101010
        );

        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 25px;
        min-height: 145px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.30);
    }

    .feature-icon {
        font-size: 30px;
        margin-bottom: 10px;
    }

    .feature-title {
        font-size: 19px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .feature-text {
        color: #999999;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Result cards */
    .result-card {
        padding: 35px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .fake-card {
        background: linear-gradient(
            135deg,
            rgba(120,0,0,0.45),
            rgba(35,0,0,0.75)
        );
        border: 1px solid #e50914;
        box-shadow: 0 0 30px rgba(229,9,20,0.18);
    }

    .real-card {
        background: linear-gradient(
            135deg,
            rgba(0,90,45,0.35),
            rgba(5,30,20,0.75)
        );
        border: 1px solid #00a86b;
        box-shadow: 0 0 30px rgba(0,168,107,0.12);
    }

    .result-label {
        font-size: 42px;
        font-weight: 900;
        letter-spacing: 3px;
    }

    .result-description {
        color: #bdbdbd;
        margin-top: 8px;
    }

    /* Section titles */
    .section-title {
        font-size: 26px;
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 18px;
    }

    .section-title span {
        color: #e50914;
    }

    /* Streamlit buttons */
    .stButton > button {
        background: #e50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
        min-height: 48px !important;
        transition: 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #b20710 !important;
        transform: scale(1.01);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #151515;
        border: 1px dashed #444444;
        border-radius: 8px;
        padding: 10px;
    }

    /* Radio */
    [data-testid="stRadio"] label {
        color: #dddddd !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #171717;
        border: 1px solid #292929;
        padding: 15px;
        border-radius: 7px;
    }

    [data-testid="stMetricLabel"] {
        color: #999999 !important;
    }

    [data-testid="stMetricValue"] {
        color: white !important;
    }

    /* Divider */
    hr {
        border-color: #292929 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666666;
        font-size: 13px;
        padding: 25px;
        margin-top: 40px;
        border-top: 1px solid #222222;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "deeptrace_model_scripted.pt"

CLASS_NAMES = ["fake", "real"]


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    model = torch.jit.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.eval()

    return model


model = load_model()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# IMAGE HASH
# ============================================================

def calculate_hash(uploaded_file):

    data = uploaded_file.getvalue()

    return hashlib.sha256(data).hexdigest()[:16]


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(image):

    image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    output = output.float()

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

    fake_probability = float(
        probabilities[0]
    )

    real_probability = float(
        probabilities[1]
    )

    if fake_probability >= real_probability:

        label = "FAKE"
        confidence = fake_probability

    else:

        label = "REAL"
        confidence = real_probability

    return (
        label,
        confidence,
        fake_probability,
        real_probability
    )


# ============================================================
# VIDEO FRAME EXTRACTION
# Uses FFmpeg instead of OpenCV
# ============================================================

def extract_video_frames(
    video_path,
    max_frames=12
):

    frame_dir = tempfile.mkdtemp()

    output_pattern = os.path.join(
        frame_dir,
        "frame_%03d.jpg"
    )

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

    try:

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )

    except Exception:

        return []

    frames = sorted(
        glob.glob(
            os.path.join(
                frame_dir,
                "*.jpg"
            )
        )
    )

    return frames


# ============================================================
# VIDEO PREDICTION
# ============================================================

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

            image = Image.open(
                frame_path
            ).convert("RGB")

            (
                label,
                confidence,
                fake_prob,
                real_prob
            ) = predict_image(image)

            fake_scores.append(
                fake_prob
            )

            real_scores.append(
                real_prob
            )

        except Exception:

            continue

    if not fake_scores:

        return None

    avg_fake = sum(fake_scores) / len(
        fake_scores
    )

    avg_real = sum(real_scores) / len(
        real_scores
    )

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
# HEADER
# ============================================================

st.markdown("""
<div class="netflix-header">

    <div>
        <div class="brand">
            DEEP-TRACE
        </div>

        <div class="tagline">
            AI-POWERED MEDIA AUTHENTICITY ANALYSIS
        </div>
    </div>

    <div class="nav-badge">
        AI DETECTION
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        Is it <span>REAL</span> or <span>FAKE?</span>
    </div>

    <div class="hero-text">
        Upload an image or video and let DEEP-TRACE
        analyze visual content using a trained deepfake
        detection model.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL STATUS
# ============================================================

if model is None:

    st.error(
        "❌ Trained model not found."
    )

    st.info(
        "Upload deeptrace_model_scripted.pt "
        "to the GitHub repository beside app.py."
    )

    st.stop()

else:

    st.success(
        "✓ DEEP-TRACE AI MODEL ONLINE"
    )


# ============================================================
# FEATURES
# ============================================================

st.markdown(
    '<div class="section-title">'
    'What DEEP-TRACE can <span>analyze</span>'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="feature-card">

        <div class="feature-icon">
            🖼️
        </div>

        <div class="feature-title">
            Image Detection
        </div>

        <div class="feature-text">
            Analyze uploaded images and estimate
            whether the visual content is real or
            synthetically manipulated.
        </div>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="feature-card">

        <div class="feature-icon">
            🎬
        </div>

        <div class="feature-title">
            Video Detection
        </div>

        <div class="feature-text">
            Sample frames from uploaded videos
            and combine frame-level predictions
            into an overall result.
        </div>

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="feature-card">

        <div class="feature-icon">
            📊
        </div>

        <div class="feature-title">
            Probability Analysis
        </div>

        <div class="feature-text">
            View confidence together with separate
            fake and real probability scores.
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Choose your <span>media</span>'
    '</div>',
    unsafe_allow_html=True
)

input_type = st.radio(
    "Media type",
    ["Image", "Video"],
    horizontal=True
)


# ============================================================
# IMAGE MODE
# ============================================================

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="image_upload"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded media",
            use_container_width=True
        )

        file_hash = calculate_hash(
            uploaded_file
        )

        st.caption(
            f"Media ID: {file_hash}"
        )

        st.write("")

        if st.button(
            "🔍 ANALYZE IMAGE",
            use_container_width=True
        ):

            with st.spinner(
                "DEEP-TRACE is analyzing the image..."
            ):

                try:

                    (
                        label,
                        confidence,
                        fake_probability,
                        real_probability
                    ) = predict_image(image)

                    st.markdown(
                        '<div class="section-title">'
                        'Detection <span>Result</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    if label == "FAKE":

                        st.markdown(
                            f"""
                            <div class="result-card fake-card">

                                <div class="result-label">
                                    🚨 FAKE
                                </div>

                                <div class="result-description">
                                    DEEP-TRACE detected patterns
                                    associated with manipulated media.
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            f"""
                            <div class="result-card real-card">

                                <div class="result-label">
                                    ✅ REAL
                                </div>

                                <div class="result-description">
                                    DEEP-TRACE detected patterns
                                    associated with authentic media.
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Confidence",
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

                    st.write("")

                    st.progress(
                        min(
                            max(
                                float(confidence),
                                0.0
                            ),
                            1.0
                        )
                    )

                    st.caption(
                        "Prediction confidence"
                    )

                except Exception as e:

                    st.error(
                        "Prediction failed."
                    )

                    st.code(
                        str(e)
                    )


# ============================================================
# VIDEO MODE
# ============================================================

else:

    uploaded_file = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv"
        ],
        key="video_upload"
    )

    if uploaded_file is not None:

        st.video(
            uploaded_file
        )

        file_hash = calculate_hash(
            uploaded_file
        )

        st.caption(
            f"Media ID: {file_hash}"
        )

        st.write("")

        if st.button(
            "🎬 ANALYZE VIDEO",
            use_container_width=True
        ):

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temp_path = temp_file.name

                with st.spinner(
                    "Extracting frames and analyzing video..."
                ):

                    result = predict_video(
                        temp_path
                    )

                if result is None:

                    st.error(
                        "Unable to extract video frames. "
                        "Please try an MP4 video."
                    )

                else:

                    (
                        label,
                        confidence,
                        fake_probability,
                        real_probability,
                        frame_count
                    ) = result

                    st.markdown(
                        '<div class="section-title">'
                        'Video Detection <span>Result</span>'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    if label == "FAKE":

                        st.markdown(
                            """
                            <div class="result-card fake-card">

                                <div class="result-label">
                                    🚨 FAKE
                                </div>

                                <div class="result-description">
                                    The analyzed video frames show
                                    stronger evidence of manipulation.
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            """
                            <div class="result-card real-card">

                                <div class="result-label">
                                    ✅ REAL
                                </div>

                                <div class="result-description">
                                    The analyzed video frames show
                                    stronger evidence of authentic media.
                                </div>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%"
                        )

                    with col2:

                        st.metric(
                            "Fake Probability",
                            f"{fake_probability * 100:.2f}%"
                        )

                    with col3:

                        st.metric(
                            "Frames Analyzed",
                            frame_count
                        )

                    st.metric(
                        "Real Probability",
                        f"{real_probability * 100:.2f}%"
                    )

                    st.write("")

                    st.progress(
                        min(
                            max(
                                float(confidence),
                                0.0
                            ),
                            1.0
                        )
                    )

                    st.caption(
                        "Video result is calculated by "
                        "aggregating predictions from sampled frames."
                    )

            except Exception as e:

                st.error(
                    "Video analysis failed."
                )

                st.code(
                    str(e)
                )

            finally:

                if (
                    temp_path is not None
                    and os.path.exists(temp_path)
                ):

                    os.remove(
                        temp_path
                    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'About <span>DEEP-TRACE</span>'
    '</div>',
    unsafe_allow_html=True
)

info1, info2 = st.columns(2)

with info1:

    st.markdown("""
    **Detection pipeline**

    1. Media uploaded  
    2. Image/frame preprocessing  
    3. Trained neural network inference  
    4. Fake/real probability calculation  
    5. Final classification  
    """)

with info2:

    st.markdown("""
    **Supported analysis**

    • Image deepfake detection  
    • Video frame analysis  
    • Fake probability  
    • Real probability  
    • Confidence score  
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    DEEP-TRACE • AI-Powered Deepfake Detection

    <br><br>

    Built using PyTorch and a FaceForensics++-based
    training dataset.

</div>
""", unsafe_allow_html=True)
