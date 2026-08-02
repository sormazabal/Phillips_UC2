"""Helpers for the Streamlit manual-correction mode: fabric.js <-> device dict,
and freehand strokes -> add/erase mask arrays. No streamlit import here so this
is testable standalone (see __main__ self-check below).
"""
import numpy as np
import cv2

# ponytail: fixed BGR palette indexed by class, no colormap dependency
# ponytail: PALETTE[2] round-trips to pure blue (#0000ff), same as ERASE_COLOR below,
# which would make that class's box strokes indistinguishable from erase brush strokes
# in strokes_to_masks. Harmless today (DEFAULT_DET_CLASSES has one class); if det_classes
# ever grows to 3+, reshuffle PALETTE so no entry hex-round-trips to pure red or pure blue.
PALETTE = [
    (0, 255, 0), (0, 165, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
]

ADD_COLOR = "#ff0000"
ERASE_COLOR = "#0000ff"


def class_colors(det_classes):
    """{class_name: '#rrggbb'} from PALETTE, cycling if there are more classes than colors."""
    colors = {}
    for i, name in enumerate(det_classes):
        b, g, r = PALETTE[i % len(PALETTE)]
        colors[name] = f"#{r:02x}{g:02x}{b:02x}"
    return colors


def devices_to_fabric(devices, scale):
    """AI detections -> canvas initial_drawing. One unfilled rect per device."""
    objects = []
    for d in devices:
        x1, y1, x2, y2 = d["bounding_box"]
        color = d.get("_color", "#00ff00")
        objects.append({
            "type": "rect",
            "left": x1 * scale,
            "top": y1 * scale,
            "width": (x2 - x1) * scale,
            "height": (y2 - y1) * scale,
            "scaleX": 1,
            "scaleY": 1,
            "angle": 0,
            "fill": "rgba(0,0,0,0)",
            "stroke": color,
            "strokeWidth": 2,
        })
    return {"objects": objects, "background": ""}


def _rect_bbox(obj, scale):
    """Axis-aligned bounding box (in original-image coords) of a possibly-rotated rect."""
    w = obj.get("width", 0) * obj.get("scaleX", 1)
    h = obj.get("height", 0) * obj.get("scaleY", 1)
    left, top = obj.get("left", 0), obj.get("top", 0)
    angle = np.deg2rad(obj.get("angle", 0) or 0)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = corners @ np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    rot += [left, top]
    x_min, y_min = rot.min(axis=0) / scale
    x_max, y_max = rot.max(axis=0) / scale
    return float(x_min), float(y_min), float(x_max), float(y_max)


def fabric_to_devices(objects, scale, ai_devices, det_classes, colors):
    """Canvas rects -> device dicts matching decode_detections' schema.
    A rect that still matches an AI detection's rounded box + class keeps that
    detection's confidence/severity/instance_id and source='model'; anything
    else (moved, resized, or newly drawn) is source='manual'.
    """
    color_to_class = {v: k for k, v in colors.items()}
    ai_by_key = {}
    for d in ai_devices:
        x1, y1, x2, y2 = [round(v) for v in d["bounding_box"]]
        ai_by_key[(x1, y1, x2, y2, d["device_class"])] = d

    devices = []
    instance_count = 1
    for obj in objects:
        if obj.get("type") != "rect":
            continue
        x1, y1, x2, y2 = _rect_bbox(obj, scale)
        x1, x2 = sorted((round(x1, 2), round(x2, 2)))
        y1, y2 = sorted((round(y1, 2), round(y2, 2)))
        stroke = (obj.get("stroke") or "").lower()
        device_class = color_to_class.get(stroke, det_classes[0])

        key = (round(x1), round(y1), round(x2), round(y2), device_class)
        ai_match = ai_by_key.get(key)

        if ai_match:
            entry = dict(ai_match)
            entry["bounding_box"] = [x1, y1, x2, y2]
            entry["source"] = "model"
        else:
            entry = {
                "device_class": device_class,
                "instance_id": f"target_{instance_count:02d}",
                "bounding_box": [x1, y1, x2, y2],
                "detection_confidence": None,
                "source": "manual",
            }
            if device_class == "coronary_stenosis":
                entry["severity"] = None
            else:
                entry["device_state"] = None
        devices.append(entry)
        instance_count += 1
    return devices


def strokes_to_masks(image_data, orig_h, orig_w):
    """Canvas RGBA drawing layer -> (add_mask, erase_mask) at original resolution.
    Add strokes are pure red, erase strokes are pure blue; anything else (e.g. green
    box outlines) is ignored by the channel-dominance filter.
    """
    if image_data is None:
        z = np.zeros((orig_h, orig_w), dtype=np.uint8)
        return z, z.copy()

    rgba = np.asarray(image_data)
    r, g, b, a = rgba[..., 0].astype(int), rgba[..., 1].astype(int), rgba[..., 2].astype(int), rgba[..., 3]
    drawn = a > 0
    add = drawn & (r > 150) & (g < 80) & (b < 80)
    erase = drawn & (b > 150) & (g < 80) & (r < 80)

    add = cv2.resize(add.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    erase = cv2.resize(erase.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return add, erase


def _demo():
    colors = class_colors(["coronary_stenosis", "device"])
    assert colors["coronary_stenosis"] == "#00ff00"

    ai_devices = [{
        "device_class": "coronary_stenosis", "instance_id": "target_01",
        "bounding_box": [10.0, 20.0, 110.0, 220.0], "detection_confidence": 0.9, "severity": "high",
    }]
    for d in ai_devices:
        d["_color"] = colors[d["device_class"]]

    scale = 0.5
    fabric = devices_to_fabric(ai_devices, scale)
    assert fabric["objects"][0]["left"] == 5.0 and fabric["objects"][0]["width"] == 50.0

    # unchanged rect round-trips as source=model, keeps confidence
    out = fabric_to_devices(fabric["objects"], scale, ai_devices, ["coronary_stenosis", "device"], colors)
    assert len(out) == 1 and out[0]["source"] == "model" and out[0]["detection_confidence"] == 0.9
    assert out[0]["bounding_box"] == [10.0, 20.0, 110.0, 220.0]

    # moved rect -> manual, drops confidence
    moved = [dict(fabric["objects"][0], left=6.0)]
    out2 = fabric_to_devices(moved, scale, ai_devices, ["coronary_stenosis", "device"], colors)
    assert out2[0]["source"] == "manual" and out2[0]["detection_confidence"] is None

    # scaled + rotated rect -> correct axis-aligned bbox (90deg rotation swaps w/h)
    rotated = [{
        "type": "rect", "left": 0, "top": 0, "width": 40, "height": 10,
        "scaleX": 1, "scaleY": 1, "angle": 90, "stroke": colors["device"],
    }]
    out3 = fabric_to_devices(rotated, 1.0, [], ["coronary_stenosis", "device"], colors)
    x1, y1, x2, y2 = out3[0]["bounding_box"]
    assert (round(x2 - x1), round(y2 - y1)) == (10, 40)
    assert out3[0]["device_class"] == "device" and out3[0]["source"] == "manual"

    # stroke splitting: red=add, blue=erase, green ignored
    img = np.zeros((4, 4, 4), dtype=np.uint8)
    img[0, 0] = [200, 0, 0, 255]   # add
    img[1, 1] = [0, 0, 200, 255]  # erase
    img[2, 2] = [0, 255, 0, 255]  # box outline color, ignored
    add, erase = strokes_to_masks(img, 4, 4)
    assert add[0, 0] == 1 and add.sum() == 1
    assert erase[1, 1] == 1 and erase.sum() == 1

    print("manual_edit self-check OK")


if __name__ == "__main__":
    _demo()
