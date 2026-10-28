"""Phase 4: a synthetic burned-in text benchmark (we know exactly where every piece of text is).

The MIDI-B answer key checks only 35 burned-in text regions, which is too few to measure a method.
So we take real images from the validation set and write text into the pixels ourselves:
- PHI text built from the patient's own identity (name, ID, date of birth in different formats, hospital acronym)
- harmless text that must stay (side markers, views, scale, window values)
For every piece of text we save its box and whether it is PHI. Then any redaction method can be scored exactly.
"""

import json
import os
import random

import cv2
import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian

from fpdeid.fingerprint import acronym

HARMLESS_TEXTS = ["L", "R", "AP", "PA", "LAT", "PORTABLE", "UPRIGHT", "SUPINE", "AXIAL", "5 mm", "10 cm",
                  "W:400 L:40", "W:1500 L:-600", "CC", "MLO", "ERECT", "kVp 120", "SCALE 1:1"]


def phi_texts(ds, rng):
    """Build 1-3 PHI strings from the patient's own header values (like a real burned-in header line)."""
    texts = []
    name = str(ds.get("PatientName", "")).replace("^", " ").strip()
    patient_id = str(ds.get("PatientID", "")).strip()
    birth = str(ds.get("PatientBirthDate", "")).strip()
    study = str(ds.get("StudyDate", "")).strip()
    institution = str(ds.get("InstitutionName", "")).strip()

    if name != "":
        sex = str(ds.get("PatientSex", "U")) or "U"
        line = name.upper()
        if rng.random() < 0.5 and len(study) == 8:
            line = line + " [" + sex + "] " + study[4:6] + "." + study[6:8] + "." + study[:4]
        texts.append(("name", line))
    if patient_id != "" and rng.random() < 0.7:
        texts.append(("id", "ID: " + patient_id))
    if len(birth) == 8 and rng.random() < 0.7:
        formats = [birth[4:6] + "." + birth[6:8] + "." + birth[:4],
                   birth[4:6] + "/" + birth[6:8] + "/" + birth[:4],
                   birth[:4] + "-" + birth[4:6] + "-" + birth[6:8]]
        texts.append(("date", "DOB: " + rng.choice(formats)))
    if institution != "" and rng.random() < 0.4:
        short = acronym(institution)
        texts.append(("institution", short if short != "" and rng.random() < 0.5 else institution.upper()))
    return texts


def to_display(pixels):
    """Any pixel array -> uint8 grey image for drawing/OCR (first frame, first channel stack kept for RGB)."""
    image = pixels.astype(np.float32)
    low = np.percentile(image, 1)
    high = np.percentile(image, 99)
    if high <= low:
        high = low + 1
    image = np.clip((image - low) / (high - low), 0, 1)
    return (image * 255).astype(np.uint8)


def draw_text(pixels, text, x, y, scale, value):
    """Write text into the real pixel array (any dtype). Returns the box [x0, y0, x1, y1]."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(scale * 2)))
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
    cv2.putText(mask, text, (x, y + height), font, scale, 255, thickness, cv2.LINE_AA)
    on = mask > 127
    if pixels.ndim == 3:
        pixels[on, :] = value
    else:
        pixels[on] = value
    box = [x, y, x + width, y + height + baseline]
    return box


def place(boxes, width, height, box_w, box_h, rng):
    """Pick a free spot near a corner or the top/bottom band, not overlapping earlier boxes."""
    margin = int(0.02 * width)
    for attempt in range(30):
        corner = rng.choice(["tl", "tr", "bl", "br", "top", "bottom"])
        if corner in ["tl", "bl"]:
            x = margin + rng.randint(0, max(1, width // 10))
        elif corner in ["tr", "br"]:
            x = width - box_w - margin - rng.randint(0, max(1, width // 10))
        else:
            x = rng.randint(margin, max(margin + 1, width - box_w - margin))
        if corner in ["tl", "tr", "top"]:
            y = margin + rng.randint(0, max(1, height // 8))
        else:
            y = height - box_h - margin - rng.randint(0, max(1, height // 8))
        x = max(0, min(x, width - box_w - 1))
        y = max(0, min(y, height - box_h - 1))
        new_box = [x, y, x + box_w, y + box_h]
        overlaps = False
        for other in boxes:
            if not (new_box[2] < other[0] or new_box[0] > other[2] or new_box[3] < other[1] or new_box[1] > other[3]):
                overlaps = True
        if not overlaps:
            return x, y
    return None


def make_one(job):
    """job = (input_path, output_path, seed). Returns a list of label dicts (or [] if the image was skipped)."""
    input_path, output_path, seed = job
    rng = random.Random(seed)
    ds = pydicom.dcmread(input_path)
    try:
        pixels = ds.pixel_array.copy()
    except Exception:
        return []
    photometric = str(ds.get("PhotometricInterpretation", ""))
    if photometric.startswith("YBR"):
        from pydicom.pixels import convert_color_space
        pixels = convert_color_space(pixels, photometric, "RGB")
        ds.PhotometricInterpretation = "RGB"
    if int(ds.get("NumberOfFrames", 1) or 1) > 1:
        return []                                   # keep the benchmark simple: single-frame images only
    height, width = pixels.shape[0], pixels.shape[1]
    if height < 128 or width < 128:
        return []

    if pixels.ndim == 3:
        bright = int(np.iinfo(pixels.dtype).max) if pixels.dtype.kind in "ui" else 255
        dark = 0
    else:
        bright = int(np.percentile(pixels, 99.5))
        dark = int(pixels.min())

    texts = phi_texts(ds, rng)
    for i in range(rng.randint(1, 3)):
        texts.append(("harmless", rng.choice(HARMLESS_TEXTS)))
    rng.shuffle(texts)

    labels = []
    boxes = []
    base_scale = max(0.4, height / 700.0)
    for kind, text in texts:
        scale = base_scale * rng.uniform(0.7, 1.3)
        thickness = max(1, int(round(scale * 2)))
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        spot = place(boxes, width, height, text_w, text_h + baseline, rng)
        if spot is None:
            continue
        value = bright if rng.random() < 0.8 else dark
        box = draw_text(pixels, text, spot[0], spot[1], scale, value)
        boxes.append(box)
        labels.append({"text": text, "kind": kind, "phi": kind != "harmless", "box": box})

    # write the image back uncompressed
    ds.PixelData = pixels.tobytes()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    if pixels.ndim == 3:
        ds.PlanarConfiguration = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.save_as(output_path, enforce_file_format=False)

    for label in labels:
        label["file"] = output_path
        label["source"] = input_path
        label["patient"] = str(ds.get("PatientID", ""))
        label["modality"] = str(ds.get("Modality", ""))
    return labels


def save_labels(labels, path):
    with open(path, "w") as f:
        json.dump(labels, f, indent=1)
