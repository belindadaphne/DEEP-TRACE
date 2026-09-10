import streamlit as st
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
# MODEL
# ============================================================

MODEL_NAME = (
    "KoreaPeter/"
    "ms-eff-gcvit-deepfake-b0-ff-plus-plus"
)


# ============================================================
# LOAD IMAGE MODEL
# ============================================================

@st.cache_resource
def load_image_model():

    return pipeline(
        "image-classification",
        model=MODEL_NAME,
        trust_remote_code=True
    )


# ============================================================
# LOAD VIDEO MODEL
# ============================================================

@st.cache_resource
def load_video_model():

    return pipeline(
        "video-classification",
        model=MODEL_NAME,
        trust_remote_code=True
    )


# ============================================================
# SHA256
# ============================================================

def sha256_bytes(data):

    return hashlib.sha256(data).hexdigest()


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
# NORMALIZE MODEL RESULT
# ============================================================

def calculate_scores(results):

    """
    Converts model output into:

        real_score
        fake_score
        verdict
        confidence

    The KoreaPeter model uses:

        real = 0
        fake = 1

    and returns labels named "real" / "fake".
    """

    real_score = 0.0
    fake_score = 0.0

    if results is None:
        results = []

    # Make sure a single dictionary also works
    if isinstance(results, dict):
        results = [results]

    for item in results:

        if not isinstance(item, dict):
            continue

        label = str(
            item.get("label", "")
        ).strip().lower()

        score = float(
            item.get("score", 0.0)
        )

        # Exact model labels
        if label == "real":

            real_score = max(
                real_score,
                score
            )

        elif label == "fake":

            fake_score = max(
                fake_score,
                score
            )

        # Extra protection for LABEL_0 / LABEL_1
        elif label in ["label_0", "0"]:

            real_score = max(
                real_score,
                score
            )

        elif label in ["label_1", "1"]:

            fake_score = max(
                fake_score,
                score
            )

    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    if fake_score > real_score:

        verdict = "DEEPFAKE"
        confidence = fake_score

    else:

        verdict = "REAL"
        confidence = real_score

    return {
        "real_score": real_score,
        "fake_score": fake_score,
        "verdict": verdict,
        "confidence": confidence
    }


# ============================================================
# IMAGE PREDICTION
# ============================================================

def run_image_prediction(image):

    detector = load_image_model()

    results = detector(
        image,
        top_k=2
    )

    scores = calculate_scores(results)

    scores["raw_results"] = results

    return scores


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(path):

    # Import only when video is used
    import cv2

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():

        return {
            "frames": 0,
            "fps": 0,
            "width": 0,
            "height": 0,
            "duration": 0
        }

    fps = cap.get(
        cv2.CAP_PROP_FPS
    ) or 0

    frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        ) or 0
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        ) or 0
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        ) or 0
    )

    duration = (
        frames / fps
        if fps > 0
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

    real_score = 0.0
    fake_score = 0.0
    frame_scores = []

    if isinstance(results, dict):

        results = [results]

    for item in results:

        if not isinstance(item, dict):
            continue

        label = str(
            item.get("label", "")
        ).strip().lower()

        score = float(
            item.get("score", 0.0)
        )

        # Model labels
        if label == "real":

            real_score = max(
                real_score,
                score
            )

        elif label == "fake":

            fake_score = max(
                fake_score,
                score
            )

        # Fallback label handling
        elif label in ["label_0", "0"]:

            real_score = max(
                real_score,
                score
            )

        elif label in ["label_1", "1"]:

            fake_score = max(
                fake_score,
                score
            )

        # Frame-level evidence
        if "frame_scores" in item:

            frame_scores = item[
                "frame_scores"
            ]

    if fake_score > real_score:

        verdict = "DEEPFAKE"
        confidence = fake_score

    else:

        verdict = "REAL"
        confidence = real_score

    return {

        "verdict": verdict,

        "confidence": confidence,

        "real_score": real_score,

        "fake_score": fake_score,

        "frame_scores": frame_scores,

        "raw_results": results

    }


# ============================================================
# RESULT DISPLAY
# ============================================================

def display_result(
    media_type,
    prediction
):

    verdict = prediction["verdict"]

    confidence = prediction["confidence"]

    real_score = prediction["real_score"]

    fake_score = prediction["fake_score"]

    if verdict == "REAL":

        result_class = "result-real"

    else:

        result_class = "result-fake"

    st.markdown(
        '<div class="section-title">'
        'RESULT'
        '</div>',
        unsafe_allow_html=True
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
        DEEPTRACE {media_type.upper()} VERDICT
        </div>

        <div class="{result_class}">
        {verdict}
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
            f"{real_score * 100:.2f}%"
        )

    with c2:

        st.metric(
            "DEEPFAKE SCORE",
            f"{fake_score * 100:.2f}%"
        )


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
        DEEPTRACE currently supports AI-based
        deepfake analysis for images and videos.

        The system compares REAL and DEEPFAKE probabilities
        and presents a model-based forensic verdict.

        Future modules will include audio analysis,
        speech transcription, lip-sync analysis,
        explainable AI, multimodal evidence correlation
        and forensic report generation.
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
        "deepfake analysis."
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

                image_bytes = (
                    uploaded_image.getvalue()
                )

                image_hash = sha256_bytes(
                    image_bytes
                )

                st.session_state.image_analysis = {

                    "name":
                        uploaded_image.name,

                    "size_mb":
                        len(image_bytes)
                        / (1024 * 1024),

                    "hash":
                        image_hash,

                    "prediction":
                        prediction

                }

                st.success(
                    "Image analysis completed."
                )

                display_result(
                    "IMAGE",
                    prediction
                )

                st.markdown(
                    '<div class="section-title">'
                    'FORENSIC INFORMATION'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    f"**File:** "
                    f"{uploaded_image.name}"
                )

                st.write(
                    f"**SHA-256:** "
                    f"`{image_hash}`"
                )

                st.write(
                    f"**Image size:** "
                    f"{image.width} × "
                    f"{image.height}"
                )

                st.write(
                    f"**Model:** "
                    f"`{MODEL_NAME}`"
                )

                st.warning(
                    "This is an AI model prediction. "
                    "It should be interpreted together with "
                    "other forensic evidence."
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

    st.write(
        "Upload a short video to perform "
        "AI-based deepfake analysis."
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

            try:

                with st.spinner(
                    "Reading video information..."
                ):

                    info = get_video_info(
                        str(video_path)
                    )

                file_hash = sha256_file(
                    str(video_path)
                )

                with st.spinner(
                    "DeepTrace is analyzing "
                    "video frames..."
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

                display_result(
                    "VIDEO",
                    prediction
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
                    f"**File:** "
                    f"{uploaded.name}"
                )

                st.write(
                    f"**SHA-256:** "
                    f"`{file_hash}`"
                )

                st.write(
                    f"**File size:** "
                    f"{video_path.stat().st_size / (1024 * 1024):.2f} MB"
                )

                st.write(
                    f"**Model:** "
                    f"`{MODEL_NAME}`"
                )

                st.write(
                    f"**Frames analyzed:** "
                    f"20"
                )

                # ------------------------------------------------
                # FRAME EVIDENCE
                # ------------------------------------------------

                frame_scores = prediction.get(
                    "frame_scores",
                    []
                )

                if frame_scores:

                    st.markdown(
                        '<div class="section-title">'
                        'FRAME-LEVEL EVIDENCE'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        "Higher values indicate stronger "
                        "deepfake evidence for the sampled frame."
                    )

                    try:

                        st.line_chart(
                            frame_scores
                        )

                    except Exception:

                        pass

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
                f"{image_prediction['verdict']}"
            )

            st.write(
                f"**Confidence:** "
                f"{image_prediction['confidence'] * 100:.2f}%"
            )

            st.write(
                f"**Real score:** "
                f"{image_prediction['real_score'] * 100:.2f}%"
            )

            st.write(
                f"**Deepfake score:** "
                f"{image_prediction['fake_score'] * 100:.2f}%"
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
                f"{video_prediction['verdict']}"
            )

            st.write(
                f"**Confidence:** "
                f"{video_prediction['confidence'] * 100:.2f}%"
            )

            st.write(
                f"**Real score:** "
                f"{video_prediction['real_score'] * 100:.2f}%"
            )

            st.write(
                f"**Deepfake score:** "
                f"{video_prediction['fake_score'] * 100:.2f}%"
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

            ✅ Real vs Deepfake probability

            ✅ Frame-level video evidence

            ⏳ Audio deepfake analysis

            ⏳ Speech transcription

            ⏳ Lip-sync analysis

            ⏳ Explainable AI

            ⏳ Multimodal evidence correlation

            ⏳ Forensic report generation
            """
        )
