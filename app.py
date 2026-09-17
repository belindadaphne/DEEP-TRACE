import streamlit as st
import torch
from PIL import Image
import numpy as np
import tempfile
import os

# ============================================================
# DEEP-TRACE
# AI-POWERED MEDIA AUTHENTICITY ANALYSIS
# ============================================================

st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# NETFLIX-STYLE THEME
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 85% 10%, rgba(229,9,20,0.18), transparent 30%),
        linear-gradient(135deg, #050505 0%, #0b0b0b 55%, #160607 100%);
    color: white;
}

header {
    background: transparent !important;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Hide Streamlit menu */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Brand */
.brand {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -2px;
    color: white;
}

.brand span {
    color: #E50914;
}

.tagline {
    color: #aaaaaa;
    font-size: 14px;
    letter-spacing: 2px;
    margin-top: -8px;
}

/* Navigation badge */
.nav-badge {
    display: inline-block;
    background: #E50914;
    color: white;
    padding: 8px 18px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* Hero */
.hero {
    padding: 55px 0 35px 0;
}

.hero-title {
    font-size: 56px;
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -2px;
}

.hero-title span {
    color: #E50914;
}

.hero-text {
    color: #b8b8b8;
    font-size: 17px;
    max-width: 700px;
    line-height: 1.7;
}

/* Cards */
.card {
    background: rgba(25,25,25,0.92);
    border: 1px solid #292929;
    border-radius: 12px;
    padding: 28px;
    margin-top: 20px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-text {
    color: #a8a8a8;
}

/* Result */
.result-real {
    background: linear-gradient(135deg, #071f12, #0b2e1a);
    border: 1px solid #1d9b59;
    border-radius: 12px;
    padding: 30px;
    margin-top: 25px;
}

.result-fake {
    background: linear-gradient(135deg, #280708, #43090d);
    border: 1px solid #E50914;
    border-radius: 12px;
    padding: 30px;
    margin-top: 25px;
}

.result-title {
    font-size: 38px;
    font-weight: 800;
}

.result-subtitle {
    color: #cccccc;
    margin-top: 5px;
}

/* Metrics */
.metric-box {
    background: #181818;
    border: 1px solid #2d2d2d;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-size: 27px;
    font-weight: 700;
}

.metric-label {
    color: #999999;
    font-size: 12px;
    margin-top: 5px;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #151515;
    border: 1px dashed #555;
    border-radius: 10px;
    padding: 12px;
}

/* Buttons */
.stButton > button {
    background: #E50914;
    color: white;
    border: none;
    border-radius: 5px;
    font-weight: 700;
    padding: 10px 24px;
}

.stButton > button:hover {
    background: #b20710;
    color: white;
}

/* Radio */
.stRadio label {
    color: white !important;
}

hr {
    border-color: #292929;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
        <div class="brand">DEEP<span>-</span>TRACE</div>
        <div class="tagline">AI-POWERED MEDIA AUTHENTICITY ANALYSIS</div>
    </div>
    <div class="nav-badge">AI DETECTION</div>
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
        Upload an image or video and let DEEP-TRACE analyze
        the media using a trained deepfake detection model.
        Detect manipulated visual content with AI-powered analysis.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "deeptrace_model_scripted.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        model = torch.jit.load(MODEL_PATH, map_location="cpu")
        model.eval()
        return model
    except Exception as e:
        st.error("Model loading failed.")
        st.code(str(e))
        return None


model = load_model()

if model is not None:
    st.success("DEEP-TRACE trained model loaded successfully.")
else:
    st.error(
        "Trained model not found. Make sure "
        "deeptrace_model_scripted.pt is in the same folder as app.py."
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_image(image):

    # Model preprocessing
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image).astype(np.float32) / 255.0

    # Normalize approximately as standard ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std

    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        output = model(tensor)

    # Handle different model output formats
    if isinstance(output, (tuple, list)):
        output = output[0]

    # Convert to probabilities
    probabilities = torch.softmax(output, dim=1)[0]

    # Your training output showed:
    # CLASS ORDER = ['fake', 'real']
    fake_probability = float(probabilities[0])
    real_probability = float(probabilities[1])

    if fake_probability > real_probability:
        verdict = "FAKE"
        confidence = fake_probability
    else:
        verdict = "REAL"
        confidence = real_probability

    return verdict, confidence, fake_probability, real_probability


# ============================================================
# INPUT TYPE
# ============================================================

st.markdown(
    '<div class="card-title">Select input type</div>',
    unsafe_allow_html=True
)

input_type = st.radio(
    "",
    ["Image", "Video"],
    horizontal=True
)

st.divider()


# ============================================================
# IMAGE DETECTION
# ============================================================

if input_type == "Image":

    st.markdown(
        '<div class="card-title">Upload an image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded media",
            use_container_width=True
        )

        if model is not None:

            with st.spinner("DEEP-TRACE is analyzing the image..."):

                try:
                    verdict, confidence, fake_prob, real_prob = predict_image(image)

                    if verdict == "FAKE":

                        st.markdown(f"""
                        <div class="result-fake">
                            <div class="result-title">⚠️ FAKE</div>
                            <div class="result-subtitle">
                                DEEP-TRACE detected characteristics
                                associated with manipulated media.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    else:

                        st.markdown(f"""
                        <div class="result-real">
                            <div class="result-title">✓ REAL</div>
                            <div class="result-subtitle">
                                DEEP-TRACE classified this media as
                                likely authentic.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("### Analysis")

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">
                                {confidence * 100:.1f}%
                            </div>
                            <div class="metric-label">
                                CONFIDENCE
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with c2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">
                                {real_prob * 100:.1f}%
                            </div>
                            <div class="metric-label">
                                REAL SCORE
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with c3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">
                                {fake_prob * 100:.1f}%
                            </div>
                            <div class="metric-label">
                                FAKE SCORE
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:

                    st.error("Prediction error.")
                    st.code(str(e))


# ============================================================
# VIDEO DETECTION
# ============================================================

else:

    st.markdown(
        '<div class="card-title">Upload a video</div>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if model is not None:

            st.info(
                "Video analysis samples frames from the uploaded video "
                "and combines their predictions."
            )

            if st.button("🔍 ANALYZE VIDEO"):

                try:

                    # Save uploaded video temporarily
                    suffix = os.path.splitext(
                        uploaded_video.name
                    )[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp:

                        temp.write(uploaded_video.read())
                        video_path = temp.name

                    # Use imageio for video reading
                    import imageio.v3 as iio

                    fake_scores = []
                    real_scores = []

                    frame_count = 0

                    with st.spinner(
                        "DEEP-TRACE is analyzing video frames..."
                    ):

                        for frame in iio.imiter(
                            video_path,
                            plugin="ffmpeg"
                        ):

                            # Analyze every ~15th frame
                            if frame_count % 15 == 0:

                                frame_image = Image.fromarray(
                                    frame
                                ).convert("RGB")

                                try:

                                    verdict, confidence, fake_p, real_p = \
                                        predict_image(frame_image)

                                    fake_scores.append(fake_p)
                                    real_scores.append(real_p)

                                except Exception:
                                    pass

                            frame_count += 1

                            # Prevent extremely long processing
                            if frame_count >= 300:
                                break

                    os.remove(video_path)

                    if len(fake_scores) == 0:

                        st.error(
                            "Could not extract frames from this video."
                        )

                    else:

                        avg_fake = float(
                            np.mean(fake_scores)
                        )

                        avg_real = float(
                            np.mean(real_scores)
                        )

                        if avg_fake > avg_real:
                            final_verdict = "FAKE"
                            final_confidence = avg_fake
                        else:
                            final_verdict = "REAL"
                            final_confidence = avg_real

                        if final_verdict == "FAKE":

                            st.markdown(f"""
                            <div class="result-fake">
                                <div class="result-title">
                                    ⚠️ FAKE
                                </div>
                                <div class="result-subtitle">
                                    Video analysis indicates possible
                                    manipulated content.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        else:

                            st.markdown(f"""
                            <div class="result-real">
                                <div class="result-title">
                                    ✓ REAL
                                </div>
                                <div class="result-subtitle">
                                    Video analysis indicates likely
                                    authentic content.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("### Video Analysis")

                        c1, c2, c3 = st.columns(3)

                        with c1:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">
                                    {final_confidence * 100:.1f}%
                                </div>
                                <div class="metric-label">
                                    CONFIDENCE
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c2:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">
                                    {avg_real * 100:.1f}%
                                </div>
                                <div class="metric-label">
                                    REAL SCORE
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c3:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">
                                    {avg_fake * 100:.1f}%
                                </div>
                                <div class="metric-label">
                                    FAKE SCORE
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.caption(
                            f"Analyzed {len(fake_scores)} video frames."
                        )

                except Exception as e:

                    st.error("Video analysis failed.")
                    st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="text-align:center;color:#666;padding:20px;">
    <b style="color:#aaa;">DEEP-TRACE</b><br>
    AI-Powered Deepfake Detection System<br>
    <small>For research and educational purposes.</small>
</div>
""", unsafe_allow_html=True)
