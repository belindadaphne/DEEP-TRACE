import os
import io
import json
import hashlib
import tempfile
from datetime import datetime, timezone

import streamlit as st
import torch
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from PIL import Image
from torchvision import transforms
=============== CONFIGURATION ===============
st.set_page_config(
page_title="DEEP-TRACE",
page_icon="🔎",
layout="wide"
)

st.title("🔎 DEEP-TRACE")
st.write("AI-assisted deepfake screening for images, videos, and audio.")

MODEL_PATH = "deeptrace_model_scripted.pt"
These class indices are assumptions from the existing app.
Verify them against the original training code.
FAKE_INDEX = 0
REAL_INDEX = 1

UNCERTAIN_MARGIN = 0.10
DEVICE = torch.device("cpu")
=============== LOAD MODEL ===============
@st.cache_resource
def load_model():
if not os.path.exists(MODEL_PATH):
raise FileNotFoundError(
f"Missing model file: "
)

loaded = torch.jit.load(
        MODEL_PATH,
        map_location=DEVICE
    )
    loaded.eval()
    return loaded

try:
model = load_model()
except Exception as exc:
st.error(f"Model loading failed: ")
st.stop()
=============== PREPROCESSING ===============
preprocess = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)
])
=============== PREDICTION ===============
def predict_image(image):
image = image.convert("RGB")
tensor = preprocess(image).unsqueeze(0)

with torch.inference_mode():
        output = model(tensor.to(DEVICE))

if isinstance(output, (tuple, list)):
        output = output[0]

if not isinstance(output, torch.Tensor):
        raise ValueError("Unsupported model output.")

    output = output.detach().float().cpu()

if output.ndim == 1:
        output = output.unsqueeze(0)

    if output.ndim != 2 or output.shape[0] <1:
        raise ValueError("Unexpected model output shape.")

    raw = output[0].numpy().astype(float)

if len(raw) != 2:
raise ValueError(
f"Expected two class outputs; received ."
)
Use existing probabilities, or convert logits to probabilities.
if (
np.all(raw >= 0)
and np.all(raw  real_score:
verdict = "FAKE"
else:
verdict = "REAL"

    return
