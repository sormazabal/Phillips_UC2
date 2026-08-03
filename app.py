import glob
import json
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from scripts.inference import decode_detections, load_model, preprocess_image
from scripts.manual_edit import (
    ADD_COLOR, ERASE_COLOR, class_colors, devices_to_fabric,
    fabric_to_devices, strokes_to_masks,
)

st.set_page_config(page_title="ARCADE XCA Dual-Task Viewer", layout="wide")
st.title("ARCADE XCA Dual-Task Viewer")
st.caption("Coronary vessel segmentation + device/stenosis localisation")

with st.sidebar:
    st.header("Model")
    checkpoints = sorted(glob.glob("checkpoints/**/*.ckpt", recursive=True))
    checkpoint_choice = st.selectbox("Checkpoint", ["(none / zero-shot)"] + checkpoints)
    checkpoint_path = None if checkpoint_choice == "(none / zero-shot)" else checkpoint_choice

    st.header("Thresholds")
    conf_thresh = st.slider("Detection confidence threshold", 0.0, 1.0, 0.5, 0.01)
    iou_thresh = st.slider("NMS IoU threshold", 0.0, 1.0, 0.5, 0.01)
    mask_thresh = st.slider("Segmentation mask threshold", 0.0, 1.0, 0.5, 0.01)

    st.header("Overlays")
    show_mask = st.checkbox("Show segmentation mask", value=True)
    show_contours = st.checkbox("Show vessel boundaries", value=True)
    show_boxes = st.checkbox("Show device/stenosis boxes/landmarks", value=True)

    st.header("Display")
    brightness = st.slider("Brightness", -100, 100, 0)
    contrast = st.slider("Contrast", 0.5, 3.0, 1.0, 0.1)

    st.header("Manual correction")
    manual_mode = st.toggle("Manual mode")
    if manual_mode:
        tool = st.radio("Tool", ["Move / resize", "New box", "Brush add", "Brush erase"])
        stroke_width = st.slider("Brush width", 1, 40, 12)
        reset_clicked = st.button("Reset to AI output")
    else:
        tool, stroke_width, reset_clicked = None, None, False


@st.cache_resource
def get_model(ckpt_path):
    return load_model(checkpoint_path=ckpt_path)


@st.cache_data
def run_inference(image_path, mtime, ckpt_path):
    """Cached forward pass. mtime busts the cache if the file on disk changes.
    Thresholds are applied outside this function so slider drags stay instant."""
    model, device = get_model(ckpt_path)
    raw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    input_tensor, (orig_h, orig_w) = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    seg_prob = torch.sigmoid(outputs["seg_logits"])[0, 0].cpu().numpy()
    det_out = outputs["det_out"].cpu().numpy()
    return raw_image, seg_prob, det_out, orig_h, orig_w


st.subheader("Input image")
uploaded = st.file_uploader("Upload an XCA frame", type=["png", "jpg", "jpeg"])
manual_path = st.text_input("...or a path under the repo (e.g. Arcade/stenosis/test/images/1.png)")

image_path = None

if uploaded is None and st.session_state.get("uploaded_tmp_path"):
    # Uploader was cleared: drop the tracked tempfile so it doesn't leak.
    stale = st.session_state.pop("uploaded_tmp_path")
    st.session_state.pop("uploaded_file_id", None)
    if os.path.exists(stale):
        os.unlink(stale)

if uploaded is not None:
    if st.session_state.get("uploaded_file_id") != uploaded.file_id:
        # Genuinely new upload (Streamlit re-returns the same UploadedFile every rerun,
        # so this only fires when the user actually picks a new file) — replace, not
        # append to, the previous tempfile.
        old_tmp = st.session_state.get("uploaded_tmp_path")
        if old_tmp and os.path.exists(old_tmp):
            os.unlink(old_tmp)
        suffix = os.path.splitext(uploaded.name)[1] or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.close()
        st.session_state["uploaded_tmp_path"] = tmp.name
        st.session_state["uploaded_file_id"] = uploaded.file_id
    image_path = st.session_state["uploaded_tmp_path"]
elif manual_path:
    image_path = manual_path

if image_path is None:
    st.info("Upload an image or enter a path to run inference.")
    st.stop()

if not os.path.exists(image_path):
    st.error(f"Image not found: {image_path}")
    st.stop()

model, device = get_model(checkpoint_path)
raw_image, seg_prob, det_out, orig_h, orig_w = run_inference(
    image_path, os.path.getmtime(image_path), checkpoint_path
)

display_image = cv2.convertScaleAbs(raw_image, alpha=contrast, beta=brightness)

mask = (seg_prob > mask_thresh).astype(np.uint8)
devices = decode_detections(torch.from_numpy(det_out), orig_h, orig_w, conf_thresh, model.det_classes, iou_thresh)

colors = class_colors(model.det_classes)
for d in devices:
    d["_color"] = colors[d["device_class"]]

new_box_class = None
if manual_mode and tool == "New box":
    # Needs model.det_classes, which isn't known until after the model loads,
    # so this appears after the rest of the "Manual correction" sidebar block.
    new_box_class = st.sidebar.selectbox("New box class", model.det_classes)

if manual_mode:
    sig = (image_path, os.path.getmtime(image_path), checkpoint_path)
    if st.session_state.get("seed_sig") != sig:
        st.session_state["seed_sig"] = sig
        st.session_state["seed_devices"] = devices
        st.session_state["canvas_ver"] = st.session_state.get("canvas_ver", 0) + 1
        st.session_state["submitted_devices"] = None
        st.session_state["submitted_add_mask"] = None
        st.session_state["submitted_erase_mask"] = None
    if reset_clicked:
        st.session_state["seed_devices"] = devices
        st.session_state["canvas_ver"] = st.session_state.get("canvas_ver", 0) + 1
        st.session_state["submitted_devices"] = None
        st.session_state["submitted_add_mask"] = None
        st.session_state["submitted_erase_mask"] = None
    seed_devices = st.session_state["seed_devices"]

    canvas_w = min(orig_w, 700)
    scale = canvas_w / orig_w
    canvas_h = round(orig_h * scale)

    bg = display_image.copy()
    colored_mask = np.zeros_like(bg)
    colored_mask[mask == 1] = (255, 0, 0)
    bg = cv2.addWeighted(bg, 1.0, colored_mask, 0.4, 0)
    bg_image = Image.fromarray(bg).resize((canvas_w, canvas_h))

    mode_map = {
        "Move / resize": ("transform", "#000000"),
        "New box": ("rect", colors.get(new_box_class, "#00ff00")),
        "Brush add": ("freedraw", ADD_COLOR),
        "Brush erase": ("freedraw", ERASE_COLOR),
    }
    drawing_mode, stroke_color = mode_map[tool]

    st.caption(
        f"Editing boxes seeded at confidence threshold {conf_thresh:.2f} at the time manual mode "
        "was opened (or last reset). Moving the confidence slider now only affects future resets."
    )
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=bg_image,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode=drawing_mode,
        initial_drawing=devices_to_fabric(seed_devices, scale),
        key=f"canvas-{sig}-{st.session_state['canvas_ver']}",
    )

    canvas_touched = canvas_result is not None and canvas_result.json_data is not None
    submit_clicked = st.button(
        "Submit corrections",
        disabled=not canvas_touched,
        help=None if canvas_touched else "Draw, move, or erase something first.",
    )
    if submit_clicked:
        objects = (canvas_result.json_data or {}).get("objects", [])
        st.session_state["submitted_devices"] = fabric_to_devices(
            objects, scale, seed_devices, model.det_classes, colors
        )
        add_mask, erase_mask = strokes_to_masks(canvas_result.image_data, orig_h, orig_w)
        st.session_state["submitted_add_mask"] = add_mask
        st.session_state["submitted_erase_mask"] = erase_mask
        st.success("Corrections submitted.")

    # Preview/export only ever reflect a submitted correction, never in-progress drawing —
    # this keeps everything below static while the operator is still adjusting the canvas.
    if st.session_state.get("submitted_devices") is not None:
        devices = st.session_state["submitted_devices"]
        add_mask = st.session_state["submitted_add_mask"]
        erase_mask = st.session_state["submitted_erase_mask"]
        mask = ((mask.astype(bool) | add_mask.astype(bool)) & ~erase_mask.astype(bool)).astype(np.uint8)

overlay = display_image.copy()

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
        color_hex = colors.get(d["device_class"], "#00ff00").lstrip("#")
        color = tuple(int(color_hex[i:i + 2], 16) for i in (4, 2, 0))  # hex RGB -> BGR
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.circle(overlay, (cx, cy), 4, color, -1)
        state = d.get("severity") or d.get("device_state")
        conf = d.get("detection_confidence")
        conf_str = f" {conf:.2f}" if conf is not None else " manual"
        label = f"{d['device_class']}" + (f" {state}" if state else "") + conf_str
        cv2.putText(overlay, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

col1, col2 = st.columns(2)
with col1:
    st.image(display_image, caption="Original", use_container_width=True)
with col2:
    st.image(overlay, caption="Annotated (mask + boxes)", use_container_width=True)

st.image(mask * 255, caption="Raw segmentation mask", use_container_width=True, clamp=True)

frame_id = os.path.splitext(os.path.basename(image_path))[0]
for d in devices:
    d.pop("_color", None)
edited = manual_mode and st.session_state.get("submitted_devices") is not None
result_json = {"frame_id": frame_id, "edited": edited, "devices": devices}
st.subheader("Detections (JSON)")
st.json(result_json)
st.download_button(
    "Download JSON",
    data=json.dumps(result_json, indent=2),
    file_name=f"{frame_id}_inference.json",
    mime="application/json",
)
