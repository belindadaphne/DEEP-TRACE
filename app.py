import streamlit as st
import cv2
import hashlib
import tempfile
from pathlib import Path
from transformers import pipeline

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="DEEPTRACE",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# NETFLIX-STYLE DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 85% 15%, rgba(229,9,20,0.20), transparent 30%),
        linear-gradient(180deg, #090909 0%, #050505 100%);
    color: white;
}

header {
    background: transparent !important;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* LOGO */

.logo {
    color: #E50914;
    font-size: 27px;
    font-weight: 900;
    letter-spacing: 6px;
    margin-bottom: 20px;
}

/* HERO */

.hero {
    min-height: 420px;
    padding: 65px 55px;
    border-radius: 18px;

    background:
        linear-gradient(
            90deg,
            #050505 5%,
            rgba(5,5,5,0.88) 38%,
            rgba(5,5,5,0.15) 100%
        ),
        radial-gradient(
            circle at 82% 45%,
            rgba(229,9,20,0.75),
            rgba(60,0,0,0.35) 30%,
            #111 70%
        );

    box-shadow: 0 25px 70px rgba(0,0,0,0.65);
    margin-bottom: 35px;
}

.hero h1 {
    font-size: 70px;
    line-height: 0.95;
    font-weight: 900;
    letter-spacing: -4px;
    margin: 45px 0 20px 0;
}

.hero p {
    max-width: 650px;
    color: #d5d5d5;
    font-size: 19px;
    line-height: 1.6;
}

/* BUTTONS */

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

/* CARDS */

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
    font-size: 29px;
    font-weight: 800;
    margin-top: 12px;
}

/* RESULT */

.result-card {
    background:
        linear-gradient(135deg, #1a1a1a, #080808);
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

/* SECTION */

.section-title {
    font-size: 28px;
    font-weight: 800;
    margin-top: 40px;
    margin-bottom: 18px;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: #111;
    border: 1px dashed #444;
    border-radius: 14px;
    padding: 15px;
}

/* METRICS */

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

MODEL_NAME = "KoreaPeter/ms-eff-gcvit-deepfake-b0-ff-plus-plus"


@st.cache_resource
def load_model():
    return pipeline(
        "video-classification",
        model=MODEL_NAME,
        trust_remote_code=True
    )


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


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
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    duration = frames / fps if fps else 0

    cap.release()

    return {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration
    }


def run_prediction(video_path):

    detector = load_model()

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

        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0))

        if label == "fake":
            fake_score = score

        elif label == "real":
            real_score = score

        elif "frame_scores" in item:
            frame_scores = item["frame_scores"]

    if not frame_scores:
        frame_scores = []

    # Final model decision
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
Analyze a video with an AI deepfake detection model.
Upload your media and receive a model-based REAL or
DEEPFAKE decision with confidence scores.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(
    "",
    ["HOME", "ANALYZE", "RESULTS"],
    horizontal=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis" not in st.session_state:
    st.session_state.analysis = None


# ============================================================
# HOME
# ============================================================

if page == "HOME":

    st.markdown(
        '<div class="section-title">DEEPTRACE</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
        <div class="card-title">Detection</div>
        <div class="card-value">REAL / FAKE</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
        <div class="card-title">Analysis</div>
        <div class="card-value">AI VIDEO</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
        <div class="card-title">Engine</div>
        <div class="card-value">MS-EffGCViT</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">How it works</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a video → the model samples video frames → "
        "faces are analyzed → frame-level predictions are "
        "aggregated → DeepTrace produces the final result."
    )

    st.info(
        "The result shown by DeepTrace is the prediction produced "
        "by the trained deepfake detection model."
    )


# ============================================================
# ANALYZE
# ============================================================

elif page == "ANALYZE":

    st.markdown(
        '<div class="section-title">UPLOAD & DETECT</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload a short video for analysis."
    )

    if uploaded:

        st.video(uploaded)

        st.markdown("")

        if st.button("▶  ANALYZE VIDEO", type="primary"):

            workdir = Path(
                tempfile.mkdtemp(prefix="deeptrace_")
            )

            video_path = workdir / uploaded.name

            video_path.write_bytes(
                uploaded.getbuffer()
            )

            info = get_video_info(str(video_path))

            file_hash = sha256_file(
                str(video_path)
            )

            try:

                with st.spinner(
                    "DeepTrace is analyzing the video..."
                ):

                    prediction = run_prediction(
                        str(video_path)
                    )

                st.session_state.analysis = {
                    "name": uploaded.name,
                    "size_mb": video_path.stat().st_size / (1024 * 1024),
                    "hash": file_hash,
                    "info": info,
                    "prediction": prediction
                }

                st.success(
                    "Analysis completed."
                )

                st.session_state.page_after_analysis = True

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
        '<div class="section-title">ANALYSIS RESULT</div>',
        unsafe_allow_html=True
    )

    analysis = st.session_state.analysis

    if not analysis:

        st.warning(
            "Upload and analyze a video first."
        )

    else:

        prediction = analysis["prediction"]
        info = analysis["info"]

        verdict = prediction["verdict"]
        meaning = prediction["meaning"]
        confidence = prediction["confidence"]

        # ---------------- RESULT ----------------

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
            DEEPTRACE VERDICT
            </div>

            <div class="{result_class}">
            {verdict}
            </div>

            <div style="
                color:white;
                font-size:30px;
                font-weight:800;
            ">
            {meaning}
            </div>

            <div class="confidence">
            Model confidence: {confidence * 100:.2f}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------------- SCORES ----------------

        st.markdown(
            '<div class="section-title">MODEL SCORES</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "REAL",
                f"{prediction['real_score'] * 100:.2f}%"
            )

        with c2:

            st.metric(
                "DEEPFAKE",
                f"{prediction['fake_score'] * 100:.2f}%"
            )

        # ---------------- VIDEO ----------------

        st.markdown(
            '<div class="section-title">VIDEO</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**File:** {analysis['name']}"
        )

        # ---------------- VIDEO INFO ----------------

        st.markdown(
            '<div class="section-title">VIDEO INFORMATION</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Frames",
                info["frames"]
            )

        with c2:
            st.metric(
                "FPS",
                f"{info['fps']:.2f}"
            )

        with c3:
            st.metric(
                "Resolution",
                f"{info['width']} × {info['height']}"
            )

        with c4:
            st.metric(
                "Duration",
                f"{info['duration']:.2f}s"
            )

        # ---------------- FILE ----------------

        st.markdown(
            '<div class="section-title">FILE INFORMATION</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**File name:** {analysis['name']}"
        )

        st.write(
            f"**File size:** {analysis['size_mb']:.2f} MB"
        )

        st.write(
            f"**SHA-256:** `{analysis['hash']}`"
        )

        # ---------------- FRAME SCORES ----------------

        frame_scores = prediction.get(
            "frame_scores",
            []
        )

        if frame_scores:

            st.markdown(
                '<div class="section-title">FRAME ANALYSIS</div>',
                unsafe_allow_html=True
            )

            st.line_chart(
                frame_scores,
                height=250
            )

            st.caption(
                "Higher frame scores indicate stronger model evidence "
                "for manipulation in those sampled frames."
            )
