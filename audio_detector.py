
import io
import hashlib
import numpy as np
import soundfile as sf
import streamlit as st


def load_audio(audio_bytes, target_sample_rate=16000):
    """Load audio bytes and return mono audio plus its sample rate."""
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if audio.size == 0:
        raise ValueError("The uploaded audio file is empty.")

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio, sample_rate


def audio_file_hash(audio_bytes):
    return hashlib.sha256(audio_bytes).hexdigest()


def show_audio_upload():
    st.subheader("Audio Deepfake Detection")

    st.info(
        "Audio preprocessing is ready. A trained audio deepfake "
        "classification model must be added before real/fake predictions "
        "can be made."
    )

    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["wav", "flac"],
        key="audio_detector_upload"
    )

    if uploaded is None:
        return

    audio_bytes = uploaded.getvalue()

    try:
        audio, sample_rate = load_audio(audio_bytes)

        st.audio(audio_bytes)
        st.write(f"Sample rate: {sample_rate} Hz")
        st.write(f"Duration: {len(audio) / sample_rate:.2f} seconds")
        st.write(f"SHA-256: `{audio_file_hash(audio_bytes)}`")

        st.warning(
            "No real/fake classification has been performed yet."
        )

    except Exception as exc:
        st.error(f"Could not read audio: {exc}")
