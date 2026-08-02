import glob
import json
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch

from scripts.inference import decode_detections, load_model, preprocess_image

st.set_page_config(page_title="ARCADE XCA Dual-Task Viewer", layout="wide")
st.title("ARCADE XCA Dual-Task Viewer")
st.caption("Coronary vessel segmentation + device/stenosis localisation")

# ponytail: fixed BGR palette indexed by class, no colormap dependency
PALETTE = [
    (0, 255, 0), (0, 165, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
]

with st.sidebar:
    st.header("Model")
    checkpoints = sorted(glob.glob("checkpoints/**/*.ckpt", recursive=True))
    checkpoint_choice = st.selectbox("Checkpoint", ["(none / zero-shot)"] + checkpoints)
    checkpoint_path = None if checkpoint_choice == "(none / zero-shot)" else checkpoint_choice

    st.header("Thresholds")
    conf_thresh = st.slider("Detection confidence threshold", 0.0, 1.0, 0.5, 0.01)
    mask_thresh = st.slider("Segmentation mask threshold", 0.0, 1.0, 0.5, 0.01)

    st.header("Overlays")
    show_mask = st.checkbox("Show segmentation mask", value=True)
    show_contours = st.checkbox("Show vessel boundaries", value=True)
    show_boxes = st.checkbox("Show device/stenosis boxes/landmarks", value=True)


@st.cache_resource
def get_model(ckpt_path):
    return load_model(checkpoint_path=ckpt_path)


st.subheader("Input image")
uploaded = st.file_uploader("Upload an XCA frame", type=["png", "jpg", "jpeg"])
manual_path = st.text_input("...or a path under the repo (e.g. Arcade/stenosis/test/images/1.png)")

image_path = None
tmp_path = None
if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    tmp_path = tmp.name
    image_path = tmp_path
elif manual_path:
    image_path = manual_path

if image_path is None:
    st.info("Upload an image or enter a path to run inference.")
    st.stop()

if not os.path.exists(image_path):
    st.error(f"Image not found: {image_path}")
    st.stop()

model, device = get_model(checkpoint_path)

raw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
input_tensor, (orig_h, orig_w) = preprocess_image(image_path)
input_tensor = input_tensor.to(device)

with torch.no_grad():
    outputs = model(input_tensor)

seg_prob = torch.sigmoid(outputs["seg_logits"])[0, 0].cpu().numpy()  # [H, W] at orig_h/orig_w
mask = (seg_prob > mask_thresh).astype(np.uint8)

devices = decode_detections(outputs["det_out"], orig_h, orig_w, conf_thresh, model.det_classes)

overlay = raw_image.copy()

if show_mask:
    colored_mask = np.zeros_like(overlay)
    colored_mask[mask == 1] = (255, 0, 0)
    overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)

if show_contours:
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

if show_boxes:
    for d in devices:
        x1, y1, x2, y2 = [int(round(v)) for v in d["bounding_box"]]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        class_idx = model.det_classes.index(d["device_class"]) if d["device_class"] in model.det_classes else 0
        color = PALETTE[class_idx % len(PALETTE)]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.circle(overlay, (cx, cy), 4, color, -1)
        state = d.get("severity") or d.get("device_state")
        label = f"{d['device_class']}" + (f" {state}" if state else "") + f" {d['detection_confidence']:.2f}"
        cv2.putText(overlay, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

col1, col2 = st.columns(2)
with col1:
    st.image(raw_image, caption="Original", use_container_width=True)
with col2:
    st.image(overlay, caption="Annotated (mask + boxes)", use_container_width=True)

st.image(mask * 255, caption="Raw segmentation mask", use_container_width=True, clamp=True)

frame_id = os.path.splitext(os.path.basename(image_path))[0]
result_json = {"frame_id": frame_id, "devices": devices}
st.subheader("Detections (JSON)")
st.json(result_json)
st.download_button(
    "Download JSON",
    data=json.dumps(result_json, indent=2),
    file_name=f"{frame_id}_inference.json",
    mime="application/json",
)

if tmp_path:
    os.unlink(tmp_path)
