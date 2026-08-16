st.set_page_config(
    page_title="DEEPTRACE",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background:
        linear-gradient(rgba(0,0,0,.82), rgba(0,0,0,.96)),
        radial-gradient(circle at 80% 15%, #5b0000, transparent 35%),
        #050505;
    color: white;
}

.block-container {
    max-width: 1250px;
    padding-top: 30px;
}

.hero {
    padding: 80px 40px;
    min-height: 420px;
    border-radius: 18px;
    background:
        linear-gradient(90deg, #050505 5%, transparent 70%),
        linear-gradient(0deg, #050505 0%, transparent 55%),
        radial-gradient(circle at 75% 35%, #710000, #111 55%);
}

.logo {
    color: #e50914;
    font-size: 25px;
    font-weight: 900;
    letter-spacing: 5px;
}

.hero h1 {
    font-size: 72px;
    font-weight: 900;
    margin: 45px 0 10px 0;
    letter-spacing: -4px;
}

.hero p {
    color: #d0d0d0;
    max-width: 600px;
    font-size: 18px;
    line-height: 1.6;
}

.card {
    background: #181818;
    border-radius: 12px;
    padding: 25px;
    border: 1px solid #292929;
}

.result {
    background: linear-gradient(135deg,#191919,#080808);
    border-radius: 18px;
    padding: 45px;
    border: 1px solid #333;
}

.stButton > button {
    background: #e50914;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 800;
    min-height: 50px;
}

.stButton > button:hover {
    background: #f40612;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="logo">DEEPTRACE</div>

<div class="hero">

<h1>THE TRUTH<br>BEHIND THE FRAME.</h1>

<p>
AI-powered deepfake detection.
Upload a video and let DeepTrace
analyze whether the media is REAL or a DEEPFAKE.
</p>

</div>
""", unsafe_allow_html=True)
