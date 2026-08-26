import streamlit as st
import cv2
import hashlib
import tempfile
from pathlib import Path
from PIL import Image
from transformers import pipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DEEPTRACE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 85% 10%,
            rgba(120, 0, 20, 0.30),
            transparent 35%
        ),
        radial-gradient(
            circle at 10% 20%,
            rgba(40, 0, 100, 0.22),
            transparent 35%
        ),
        #050505;
    color: white;
}

.block-container {
    max-width: 1180px;
    padding-top: 35px;
    padding-bottom: 60px;
}

.logo {
    font-size: 27px;
    font-weight: 900;
    letter-spacing: 8px;
    color: #ffffff;
    margin-bottom: 25px;
}

.hero {
    background:
        linear-gradient(
            135deg,
            rgba(15,15,15,0.97),
            rgba(30,0,5,0.94)
        );
    border-radius: 22px;
    padding: 65px;
    margin-bottom: 40px;
    border: 1px solid #242424;
    box-shadow: 0 25px 70px rgba(0,0,0,0.65);
}

.hero h1 {
    font-size: 70px;
    line-height: 0.95;
    font-weight: 900;
    letter-spacing: -4px;
    margin: 45px 0 20px 0;
}

.hero p {
    max-width: 720px;
    color: #d5d5d5;
    font-size: 19px;
    line-height: 1.6;
}

.section-title {
    font-size: 28px;
    font-weight: 800;
    margin-top: 40px;
    margin-bottom: 18px;
}

.card {
    background: #151515;
    border: 1px solid #292929;
    border-radius: 14px;
    padding: 25px;
    min-height: 140px;
}

.card-title {
    color: #a8a8a8;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.card-value {
    color: white;
    font-size: 27px;
    font-weight: 800;
    margin-top: 12px;
}

.result-card {
    background:
        linear-gradient(
            135deg,
            #1a1a1a,
            #080808
        );
    border: 1px solid #333;
    border-radius: 18px;
    padding: 45px;
    text-align: center;
    margin-top: 25px;
}

.result-real {
    color: #31d158;
    font-size: 58px;
    font-weight: 900;
}

.result-fake {
    color: #E50914;
    font-size: 58px;
    font-weight: 900;
}

.confidence {
    color: #bdbdbd;
    font-size: 18px;
    margin-top: 10px;
}

.stButton > button {
    background: #E50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 7px !important;
    min-height: 52px;
    font-size: 16px;
    font-weight: 800;
}

.stButton > button:hover {
    background: #ff1823 !important;
}

[data-testid="stFileUploader"] {
    background: #111;
    border: 1px dashed #444;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stMetric"] {
    background: #151515;
    border: 1px solid #292929;
    border-radius: 12px;
    padding: 18px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="logo">DEEPTRACE</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL NAMES
# ============================================================

VIDEO_MODEL_NAME = (
    "KoreaPeter/"
    "ms-eff-gcvit-deepfake-b0-ff-plus-plus"
)

IMAGE_MODEL_NAME = "king1oo1/deepfake-model"


# ============================================================
# LOAD VIDEO MODEL
# ============================================================

@st.cache_resource
def load_video_model():

    return pipeline(
        "video-classification",
        model=VIDEO_MODEL_NAME,
        trust_remote_code=True
    )


# ============================================================
# LOAD IMAGE MODEL
# ============================================================

@st.cache_resource
def load_image_model():

    return pipeline(
        "image-classification",
        model=IMAGE_MODEL_NAME
    )


# ============================================================
# SHA256
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(path):

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():

        return {
            "frames": 0,
            "fps": 0,
            "width": 0,
            "height": 0,
            "duration": 0
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 0

    frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    )

    duration = (
        frames / fps
        if fps
        else 0
    )

    cap.release()

    return {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration
    }


# ============================================================
# VIDEO PREDICTION
# ============================================================

def run_video_prediction(video_path):

    detector = load_video_model()

    results = detector(
        video_path,
        num_frames=20,
        agg_mode="conf",
        return_frame_scores=True
    )

    fake_score = 0.0
    real_score = 0.0
    frame_scores = []

    for item in results:

        if not isinstance(item, dict):
            continue

        label = str(
            item.get("label", "")
        ).lower()

        score = float(
            item.get("score", 0)
        )

        if label == "fake":

            fake_score = score

        elif label == "real":

            real_score = score

        elif "frame_scores" in item:

            frame_scores = item[
                "frame_scores"
            ]

    if fake_score >= real_score:

        verdict = "FALSE"
        meaning = "DEEPFAKE"
        confidence = fake_score

    else:

        verdict = "TRUE"
        meaning = "REAL"
        confidence = real_score

    return {

        "verdict": verdict,

        "meaning": meaning,

        "confidence": confidence,

        "fake_score": fake_score,

        "real_score": real_score,

        "frame_scores": frame_scores

    }


# ============================================================
# IMAGE PREDICTION
# ============================================================

def run_image_prediction(image):

    detector = load_image_model()

    results = detector(image)

    fake_score = 0.0
    real_score = 0.0

    for item in results:

        label = str(
            item.get("label", "")
        ).lower()

        score = float(
            item.get("score", 0)
        )

        if "fake" in label or "deepfake" in label:

            fake_score = max(
                fake_score,
                score
            )

        elif "real" in label:

            real_score = max(
                real_score,
                score
            )

    if fake_score >= real_score:

        meaning = "DEEPFAKE"
        verdict = "FALSE"
        confidence = fake_score

    else:

        meaning = "REAL"
        verdict = "TRUE"
        confidence = real_score

    return {

        "verdict": verdict,

        "meaning": meaning,

        "confidence": confidence,

        "fake_score": fake_score,

        "real_score": real_score,

        "raw_results": results

    }


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div style="
color:#bbbbbb;
font-size:14px;
font-weight:700;
letter-spacing:4px;
">
AI MEDIA FORENSICS
</div>

<h1>
THE TRUTH<br>
BEHIND THE FRAME.
</h1>

<p>
DEEPTRACE is an explainable multimodal AI framework
for investigating potentially manipulated digital media.
Analyze images and videos and receive model-based
REAL or DEEPFAKE results with confidence information.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(
    "",
    [
        "HOME",
        "IMAGE",
        "VIDEO",
        "RESULTS"
    ],
    horizontal=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "video_analysis" not in st.session_state:

    st.session_state.video_analysis = None


if "image_analysis" not in st.session_state:

    st.session_state.image_analysis = None


# ============================================================
# HOME
# ============================================================

if page == "HOME":

    st.markdown(
        '<div class="section-title">'
        'DEEPTRACE'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        Media
        </div>

        <div class="card-value">
        IMAGE + VIDEO
        </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        Detection
        </div>

        <div class="card-value">
        REAL / DEEPFAKE
        </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        AI
        </div>

        <div class="card-value">
        DEEP LEARNING
        </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">'
        'Current Review 2 Development'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The current prototype supports video analysis and
        the Review 2 image-analysis module is being integrated.
        The planned system will later include audio analysis,
        transcription, lip-sync analysis, explainability,
        multimodal evidence correlation and forensic reporting.
        """
    )

    st.info(
        "DeepTrace results are model predictions and should "
        "be interpreted as forensic decision-support evidence, "
        "not as absolute proof."
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

elif page == "IMAGE":

    st.markdown(
        '<div class="section-title">'
        'IMAGE ANALYSIS'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload an image to perform AI-based "
        "deepfake image analysis."
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        help="Upload a facial or general image."
    )

    if uploaded_image:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        if st.button(
            "▶  ANALYZE IMAGE",
            type="primary"
        ):

            try:

                with st.spinner(
                    "DeepTrace is analyzing the image..."
                ):

                    prediction = (
                        run_image_prediction(
                            image
                        )
                    )

                image_hash = hashlib.sha256(
                    uploaded_image.getvalue()
                ).hexdigest()

                st.session_state.image_analysis = {

                    "name":
                        uploaded_image.name,

                    "size_mb":
                        len(
                            uploaded_image.getvalue()
                        ) / (1024 * 1024),

                    "hash":
                        image_hash,

                    "prediction":
                        prediction

                }

                st.success(
                    "Image analysis completed."
                )

                st.markdown(
                    '<div class="section-title">'
                    'RESULT'
                    '</div>',
                    unsafe_allow_html=True
                )

                meaning = prediction[
                    "meaning"
                ]

                confidence = prediction[
                    "confidence"
                ]

                result_class = (
                    "result-real"
                    if meaning == "REAL"
                    else "result-fake"
                )

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div style="
                    color:#999;
                    font-size:14px;
                    letter-spacing:4px;
                    font-weight:700;
                    ">
                    DEEPTRACE IMAGE VERDICT
                    </div>

                    <div class="{result_class}">
                    {meaning}
                    </div>

                    <div class="confidence">
                    Model confidence:
                    {confidence * 100:.2f}%
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "REAL SCORE",
                        f"{prediction['real_score'] * 100:.2f}%"
                    )

                with c2:

                    st.metric(
                        "FAKE SCORE",
                        f"{prediction['fake_score'] * 100:.2f}%"
                    )

                st.markdown(
                    '<div class="section-title">'
                    'FORENSIC INFORMATION'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    f"**File:** {uploaded_image.name}"
                )

                st.write(
                    f"**SHA-256:** `{image_hash}`"
                )

                st.write(
                    f"**Image size:** "
                    f"{image.width} × {image.height}"
                )

                st.warning(
                    "This is a model-based prediction. "
                    "It should not be treated as absolute proof "
                    "of manipulation."
                )

            except Exception as e:

                st.error(
                    "The image model could not analyze "
                    "this image."
                )

                st.exception(e)


# ============================================================
# VIDEO ANALYSIS
# ============================================================

elif page == "VIDEO":

    st.markdown(
        '<div class="section-title">'
        'VIDEO ANALYSIS'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Choose a video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv"
        ],
        help="Upload a short video for analysis."
    )

    if uploaded:

        st.video(uploaded)

        if st.button(
            "▶  ANALYZE VIDEO",
            type="primary"
        ):

            workdir = Path(
                tempfile.mkdtemp(
                    prefix="deeptrace_"
                )
            )

            video_path = (
                workdir / uploaded.name
            )

            video_path.write_bytes(
                uploaded.getbuffer()
            )

            info = get_video_info(
                str(video_path)
            )

            file_hash = sha256_file(
                str(video_path)
            )

            try:

                with st.spinner(
                    "DeepTrace is analyzing the video..."
                ):

                    prediction = (
                        run_video_prediction(
                            str(video_path)
                        )
                    )

                st.session_state.video_analysis = {

                    "name":
                        uploaded.name,

                    "size_mb":
                        video_path.stat().st_size
                        / (1024 * 1024),

                    "hash":
                        file_hash,

                    "info":
                        info,

                    "prediction":
                        prediction

                }

                st.success(
                    "Video analysis completed."
                )

                meaning = prediction[
                    "meaning"
                ]

                confidence = prediction[
                    "confidence"
                ]

                result_class = (
                    "result-real"
                    if meaning == "REAL"
                    else "result-fake"
                )

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div style="
                    color:#999;
                    font-size:14px;
                    letter-spacing:4px;
                    font-weight:700;
                    ">
                    DEEPTRACE VIDEO VERDICT
                    </div>

                    <div class="{result_class}">
                    {meaning}
                    </div>

                    <div class="confidence">
                    Model confidence:
                    {confidence * 100:.2f}%
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    st.metric(
                        "FRAMES",
                        info["frames"]
                    )

                with c2:

                    st.metric(
                        "FPS",
                        f"{info['fps']:.2f}"
                    )

                with c3:

                    st.metric(
                        "RESOLUTION",
                        f"{info['width']} × "
                        f"{info['height']}"
                    )

                with c4:

                    st.metric(
                        "DURATION",
                        f"{info['duration']:.2f}s"
                    )

                st.markdown(
                    '<div class="section-title">'
                    'FORENSIC INFORMATION'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    f"**File:** {uploaded.name}"
                )

                st.write(
                    f"**SHA-256:** `{file_hash}`"
                )

                st.write(
                    f"**File size:** "
                    f"{video_path.stat().st_size / (1024 * 1024):.2f} MB"
                )

                st.write(
                    f"**Real score:** "
                    f"{prediction['real_score'] * 100:.2f}%"
                )

                st.write(
                    f"**Deepfake score:** "
                    f"{prediction['fake_score'] * 100:.2f}%"
                )

                st.warning(
                    "The result is a model prediction. "
                    "It should be interpreted together with "
                    "other forensic evidence."
                )

            except Exception as e:

                st.error(
                    "The AI model could not analyze this video."
                )

                st.exception(e)


# ============================================================
# RESULTS
# ============================================================

elif page == "RESULTS":

    st.markdown(
        '<div class="section-title">'
        'ANALYSIS RESULTS'
        '</div>',
        unsafe_allow_html=True
    )

    has_image = (
        st.session_state.image_analysis
        is not None
    )

    has_video = (
        st.session_state.video_analysis
        is not None
    )

    if not has_image and not has_video:

        st.info(
            "Analyze an image or video first."
        )

    else:

        if has_image:

            image_data = (
                st.session_state.image_analysis
            )

            image_prediction = (
                image_data["prediction"]
            )

            st.markdown(
                "### 🖼️ IMAGE RESULT"
            )

            st.write(
                f"**File:** "
                f"{image_data['name']}"
            )

            st.write(
                f"**Result:** "
                f"{image_prediction['meaning']}"
            )

            st.write(
                f"**Confidence:** "
                f"{image_prediction['confidence'] * 100:.2f}%"
            )

        if has_video:

            video_data = (
                st.session_state.video_analysis
            )

            video_prediction = (
                video_data["prediction"]
            )

            st.markdown(
                "### 🎥 VIDEO RESULT"
            )

            st.write(
                f"**File:** "
                f"{video_data['name']}"
            )

            st.write(
                f"**Result:** "
                f"{video_prediction['meaning']}"
            )

            st.write(
                f"**Confidence:** "
                f"{video_prediction['confidence'] * 100:.2f}%"
            )

        st.markdown(
            '<div class="section-title">'
            'REVIEW 2 ROADMAP'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            ✅ Image deepfake analysis

            ✅ Video deepfake analysis

            ⏳ Audio deepfake analysis

            ⏳ Speech transcription

            ⏳ Lip-sync analysis

            ⏳ Explainable AI

            ⏳ Multimodal evidence correlation

            ⏳ Forensic report generation
            """
        )
