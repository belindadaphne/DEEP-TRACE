
import os
import io
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


# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="DEEP-TRACE",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 DEEP-TRACE")
st.caption("AI-assisted image and video deepfake analysis")

MODEL_PATH = "deeptrace_model_scripted.pt"

# Preserves the current app's assumed output order.
# Verify this against the model's training code.
FAKE_INDEX = 0
REAL_INDEX = 1

# Small score differences are reported as uncertain.
UNCERTAIN_MARGIN = 0.10

DEVICE = torch.device("cpu")


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = torch.jit.load(
        MODEL_PATH,
        map_location=DEVICE
    )
    model.eval()
    return model


try:
    model = load_model()
    st.success("Detection model loaded.")
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()


# ---------------- PREPROCESSING ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------- PREDICTION ----------------
def predict_image(image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise ValueError("Model returned an unsupported output.")

    output = output.detach().float().cpu()

    if output.ndim == 1:
        output = output.unsqueeze(0)

    if output.ndim != 2 or output.shape[0] < 1:
        raise ValueError(
            f"Unexpected model output shape: {tuple(output.shape)}"
        )

    values = output[0].numpy().astype(float)

    if len(values) != 2:
        raise ValueError(
            f"Expected 2 class outputs, got {len(values)}."
        )

    # Convert logits to scores; preserve already-normalized probabilities.
    if (
        np.all(values >= 0)
        and np.all(values <= 1)
        and np.isclose(values.sum(), 1.0, atol=1e-3)
    ):
        probabilities = values
    else:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        probabilities = exp_values / exp_values.sum()

    fake_score = float(probabilities[FAKE_INDEX])
    real_score = float(probabilities[REAL_INDEX])

    difference = abs(fake_score - real_score)

    if difference < UNCERTAIN_MARGIN:
        verdict = "UNCERTAIN"
    elif fake_score > real_score:
        verdict = "FAKE"
    else:
        verdict = "REAL"

    return {
        "verdict": verdict,
        "fake_score": fake_score,
        "real_score": real_score,
        "raw_output": values.tolist(),
        "model_scores": probabilities.tolist()
    }


# ---------------- FILE HASH ----------------
def sha256_file(data):
    return hashlib.sha256(data).hexdigest()


# ---------------- IMAGE ANALYSIS ----------------
st.header("Image analysis")

image_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    key="image_upload"
)

if image_file is not None:
    image_bytes = image_file.getvalue()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(image, caption="Uploaded image", width=350)

        if st.button("Analyze image", type="primary"):
            with st.spinner("Analyzing image..."):
                result = predict_image(image)

            st.subheader("Image result")
            st.write(f"**Verdict:** {result['verdict']}")

            col1, col2 = st.columns(2)
            col1.metric(
                "REAL score",
                f"{result['real_score'] * 100:.2f}%"
            )
            col2.metric(
                "FAKE score",
                f"{result['fake_score'] * 100:.2f}%"
            )

            with st.expander("Diagnostic model output"):
                st.write("Raw model output:", result["raw_output"])
                st.write("Model scores:", result["model_scores"])
                st.caption(
                    "Scores are model outputs, not proof of authenticity."
                )

            image_report = {
                "file_name": image_file.name,
                "file_type": "image",
                "sha256": sha256_file(image_bytes),
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "verdict": result["verdict"],
                "real_score": result["real_score"],
                "fake_score": result["fake_score"],
                "raw_model_output": result["raw_output"],
                "model_scores": result["model_scores"]
            }

            st.download_button(
                "Download forensic report (JSON)",
                data=json.dumps(
                    image_report, indent=2
                ).encode("utf-8"),
                file_name="deep_trace_image_report.json",
                mime="application/json"
            )

    except Exception as exc:
        st.error(f"Image analysis failed: {exc}")


# ---------------- VIDEO ANALYSIS ----------------
st.divider()
st.header("Video analysis")

video_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    key="video_upload"
)

if video_file is not None:
    video_bytes = video_file.getvalue()

    st.video(video_bytes)

    if st.button("Analyze video", type="primary"):
        temp_path = None
        reader = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(video_file.name)[1]
                or ".mp4",
                delete=False
            ) as temp_file:
                temp_file.write(video_bytes)
                temp_path = temp_file.name

            with st.spinner("Decoding video and analyzing frames..."):
                reader = imageio.get_reader(
                    temp_path,
                    format="ffmpeg"
                )

                metadata = reader.get_meta_data()
                fps = float(metadata.get("fps") or 0)

                rows = []
                decoded_count = 0

                # Sample up to 10 evenly spaced frames while decoding.
                # Frame spacing is approximately every 0.5 seconds.
                sample_every = max(
                    1, int(round(fps * 0.5))
                ) if fps > 0 else 15

                for frame_index, frame in enumerate(reader):
                    decoded_count += 1

                    if frame_index % sample_every != 0:
                        continue

                    if len(rows) >= 10:
                        break

                    frame_image = Image.fromarray(
                        np.asarray(frame).astype(np.uint8)
                    ).convert("RGB")

                    result = predict_image(frame_image)

                    rows.append({
                        "Frame": frame_index,
                        "Timestamp (seconds)": (
                            frame_index / fps if fps > 0 else None
                        ),
                        "Prediction": result["verdict"],
                        "REAL score (%)": round(
                            result["real_score"] * 100, 2
                        ),
                        "FAKE score (%)": round(
                            result["fake_score"] * 100, 2
                        ),
                        "Score difference (%)": round(
                            abs(
                                result["real_score"]
                                - result["fake_score"]
                            ) * 100, 2
                        )
                    })

                # If the loop stopped early after 10 samples,
                # decoded_count is the number reached, not necessarily
                # the video's total frame count.
                if not rows:
                    raise ValueError(
                        "No video frames could be analyzed."
                    )

            results_df = pd.DataFrame(rows)

            real_average = (
                results_df["REAL score (%)"].mean() / 100
            )
            fake_average = (
                results_df["FAKE score (%)"].mean() / 100
            )

            avg_difference = abs(real_average - fake_average)

            if avg_difference < UNCERTAIN_MARGIN:
                overall_verdict = "UNCERTAIN"
            elif fake_average > real_average:
                overall_verdict = "FAKE"
            else:
                overall_verdict = "REAL"

            st.subheader("Overall video result")
            st.write(f"**Verdict: {overall_verdict}**")

            col1, col2 = st.columns(2)
            col1.metric(
                "REAL score",
                f"{real_average * 100:.2f}%"
            )
            col2.metric(
                "FAKE score",
                f"{fake_average * 100:.2f}%"
            )

            with st.expander("Diagnostic model output"):
                st.write(
                    "First sampled frame raw model output:",
                    predict_image(
                        Image.fromarray(
                            np.asarray(
                                next(iter([np.zeros((224, 224, 3), dtype=np.uint8)]))
                            )
                        )
                    )["raw_output"]
                    if False else
                    "See per-frame scores in the table below."
                )
                st.write(
                    "Assumed output order:",
                    f"index {FAKE_INDEX} = FAKE, "
                    f"index {REAL_INDEX} = REAL"
                )
                st.write(
                    "Average REAL score:",
                    round(real_average, 4)
                )
                st.write(
                    "Average FAKE score:",
                    round(fake_average, 4)
                )

            st.caption(
                "Scores are relative model outputs, not calibrated "
                "probabilities or proof of authenticity."
            )

            st.write(f"**Frames analyzed:** {len(rows)}")
            st.write(f"**Frames decoded before stopping:** {decoded_count}")
            st.write(f"**Video FPS:** {fps:.3f}" if fps else
                     "**Video FPS:** unavailable")

            st.subheader("Sampled frame results")
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            report = {
                "file_name": video_file.name,
                "file_type": "video",
                "sha256": sha256_file(video_bytes),
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "video_fps": fps,
                "frames_analyzed": len(rows),
                "frames_decoded_before_stopping": decoded_count,
                "overall_verdict": overall_verdict,
                "average_real_score": real_average,
                "average_fake_score": fake_average,
                "uncertain_margin": UNCERTAIN_MARGIN,
                "assumed_class_order": {
                    str(FAKE_INDEX): "FAKE",
                    str(REAL_INDEX): "REAL"
                },
                "frame_results": rows,
                "warning": (
                    "Experimental AI output. Not proof of authenticity."
                )
            }

            st.subheader("File integrity")
            st.code(sha256_file(video_bytes))

            st.download_button(
                "Download forensic report (JSON)",
                data=json.dumps(
                    report, indent=2
                ).encode("utf-8"),
                file_name="deep_trace_video_report.json",
                mime="application/json"
            )

            st.download_button(
                "Download frame results (CSV)",
                data=results_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="deep_trace_frame_results.csv",
                mime="text/csv"
            )

        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")
            st.info(
                "Check the Streamlit app logs if the problem continues. "
                "Make sure imageio-ffmpeg is listed in requirements.txt."
            )

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


st.divider()
st.caption(
    "DEEP-TRACE is an experimental screening tool. "
    "Predictions can be wrong; verify important findings independently."
)
```
