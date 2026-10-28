"""Phase 4: find and hide burned-in text in the pixels."""

import json
import os
import re
import time

import cv2
import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
from rapidfuzz import fuzz

# words that are normal on medical images and must stay readable
MARKERS = set([
    "L", "R", "AP", "PA", "LAT", "LL", "RL", "LR", "RT", "LT", "RLAT", "LLAT", "OBL", "PORTABLE", "PORT", "UPRIGHT",
    "SUPINE", "PRONE", "ERECT", "STANDING", "SEMI", "DECUB", "CHEST", "AXIAL", "SAG", "COR", "CC", "MLO", "XCCL",
    "RCC", "LCC", "RMLO", "LMLO", "ML", "LM", "SPOT", "MAG", "MM", "CM", "KVP", "MAS", "SCALE", "W", "WL", "WW",
    "TRAUMA", "INSP", "EXP", "ABD", "PELVIS", "SKULL", "HAND", "FOOT", "KNEE", "HIP", "SPINE",
])
LABEL_WORDS = ["DOB", "D.O.B", "BIRTH", "BORN", "MRN", "ID", "NAME", "PATIENT", "ACC", "ACCESSION"]
DATE_IN_TEXT = re.compile(r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b|\b(?:19|20)\d{2}[./\-]?\d{2}[./\-]?\d{2}\b")
INITIALS = re.compile(r"^[A-Za-z]{2,4}$")
XRAY_MODALITIES = ["CR", "DX", "MG", "RG", "PX", "IO"]

READER = {}


def get_reader():
    """Load EasyOCR once (on the GPU if there is one)."""
    if "reader" not in READER:
        import os
        import easyocr
        import torch
        from fpdeid import paths
        # reuse the model files downloaded for the official scorer (no second download)
        model_folder = os.path.join(paths.PROJECT, "tools", "easyocr", "model")
        READER["reader"] = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False,
                                          model_storage_directory=model_folder)
    return READER["reader"]


# 1. Views of the image for reading

def to_gray(pixels):
    if pixels.ndim == 3:
        return pixels.astype(np.float32).max(axis=2)
    return pixels.astype(np.float32)


def window(image, low, high):
    if high <= low:
        high = low + 1
    scaled = np.clip((image - low) / (high - low), 0, 1)
    return (scaled * 255).astype(np.uint8)


def make_views(pixels):
    """Three 8-bit views: normal, bright text (top of the value range), dark text (bottom of the range, inverted)."""
    image = to_gray(pixels)
    p1, p50, p99 = np.percentile(image, [1, 50, 99])
    top = image.max()
    bottom = image.min()
    normal = window(image, p1, p99)
    bright = window(image, max(p50, p99 - 0.05 * (top - bottom)), top)
    dark = 255 - window(image, bottom, min(p50, p1 + 0.05 * (top - bottom)))
    return [normal, bright, dark]


# 2. OCR and text lines

def read_view(image8, min_confidence):
    reader = get_reader()
    found = []
    for points, text, confidence in reader.readtext(image8):
        if confidence < min_confidence or text.strip() == "":
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        found.append(([int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1], text, float(confidence)))
    return found


def read_text(image8, min_confidence=0.3):
    """OCR one 8-bit image (kept for simple uses, e.g. checking leaks)."""
    return read_view(image8, min_confidence)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def ocr_image(pixels, min_confidence=0.0, target_size=2000):
    """OCR all views (upscaled if small) and merge boxes that are found more than once."""
    height, width = pixels.shape[0], pixels.shape[1]
    scale = 1.0
    if max(height, width) < target_size:
        scale = target_size / max(height, width)
    found = []
    for view in make_views(pixels):
        if scale != 1.0:
            view = cv2.resize(view, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)
        for box, text, confidence in read_view(view, min_confidence):
            box = [int(box[0] / scale), int(box[1] / scale), int(np.ceil(box[2] / scale)), int(np.ceil(box[3] / scale))]
            found.append((box, text, confidence))
    # merge: if two boxes overlap strongly, keep the one with the higher confidence
    found.sort(key=lambda item: -item[2])
    merged = []
    for item in found:
        if all(iou(item[0], other[0]) < 0.5 for other in merged):
            merged.append(item)
    return merged


def group_lines(found):
    """Join boxes that sit on the same text line (vertical overlap, small horizontal gap)."""
    items = sorted(found, key=lambda item: (item[0][1], item[0][0]))
    lines = []
    for box, text, confidence in items:
        placed = False
        for line in lines:
            lbox = line["box"]
            height = min(box[3] - box[1], lbox[3] - lbox[1])
            vertical = min(box[3], lbox[3]) - max(box[1], lbox[1])
            gap = max(box[0] - lbox[2], lbox[0] - box[2])
            if vertical > 0.5 * height and gap < 2.0 * height:
                line["parts"].append((box, text))
                line["box"] = [min(lbox[0], box[0]), min(lbox[1], box[1]), max(lbox[2], box[2]), max(lbox[3], box[3])]
                line["confidence"] = min(line["confidence"], confidence)
                placed = True
                break
        if not placed:
            lines.append({"box": list(box), "parts": [(box, text)], "confidence": confidence})
    for line in lines:
        line["parts"].sort(key=lambda part: part[0][0])
        line["text"] = " ".join(text for box, text in line["parts"])
    return lines


# 3. Decide per line

def is_marker(text):
    words = re.findall(r"[A-Za-z]+", text.upper())
    if len(words) == 0:
        return True                      # only numbers / symbols, e.g. "5", "1:1"
    return all(word in MARKERS for word in words)


def fuzzy_fingerprint(text, fingerprint, threshold=80):
    """OCR makes small mistakes (BALDWIN -> PALDWIN). Compare each word/number with the fingerprint entries."""
    words = re.findall(r"[A-Za-z0-9]{4,}", text.upper())
    if len(words) == 0:
        return None
    for variant, category, source in fingerprint.entries:
        if category not in ["name", "id", "date", "institution"]:
            continue
        for target in re.findall(r"[A-Za-z0-9]{4,}", variant.upper()):
            for word in words:
                if abs(len(word) - len(target)) > 2:
                    continue
                if fuzz.ratio(word, target) >= threshold:
                    return category + " (" + source + ", fuzzy)"
    return None


def decide(text, fingerprint, modality="", common_codes=None, confidence=1.0):
    """Return (hide?, reason) for one line of text found in the image."""
    clean = text.strip()
    upper = clean.upper()
    if fingerprint is not None:
        spans = fingerprint.find(clean)
        if len(spans) > 0:
            start, end, category, source = spans[0]
            return True, category + " (" + source + ")"
        fuzzy = fuzzy_fingerprint(clean, fingerprint)
        if fuzzy is not None:
            return True, fuzzy
    for label in LABEL_WORDS:
        if re.search(r"(?<![A-Z])" + re.escape(label) + r"(?![A-Z])", upper):
            return True, "label word " + label
    if DATE_IN_TEXT.search(clean):
        return True, "date in the image"
    if is_marker(clean):
        return False, "radiology marker / numbers"
    # initials: ONE clean word (no spaces), read with good confidence -- OCR noise like "Lo C C" is not initials
    letters = re.sub(r"[^A-Za-z]", "", clean)
    one_word = " " not in clean
    if modality in XRAY_MODALITIES and one_word and confidence >= 0.5 and INITIALS.match(letters) \
            and len(clean) - len(letters) <= 2:
        if common_codes is not None and letters.upper() in common_codes:
            return False, "word seen in the images of many patients"
        return True, "short code that looks like initials"
    return False, "no identifier found"


# 4. Hide

def hide_boxes(pixels, boxes, photometric, margin=None):
    """Fill the boxes in the real pixel array with the 'black' value of the image."""
    if photometric == "MONOCHROME1":
        fill = pixels.max()              # in MONOCHROME1 the highest value is shown as black
    else:
        fill = pixels.min()
    height, width = pixels.shape[0], pixels.shape[1]
    for x0, y0, x1, y1 in boxes:
        pad = margin if margin is not None else max(4, int(0.2 * (y1 - y0)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(width, x1 + pad)
        y1 = min(height, y1 + pad)
        pixels[y0:y1, x0:x1] = fill
    return pixels


def band_boxes(height, width, share=0.12):
    """The 'bands' method: the top and bottom part of the image."""
    band = int(share * height)
    return [[0, 0, width, band], [0, height - band, width, height]]


def boxes_for_method(found, method, fingerprint, modality, common_codes, height, width, min_confidence=0.0):
    """found = OCR result [(box, text, conf)] -> (boxes to hide, decisions)."""
    if method == "bands":
        return band_boxes(height, width), []
    kept = [item for item in found if item[2] >= min_confidence]
    boxes = []
    decisions = []
    for line in group_lines(kept):
        if method == "all_text":
            hide, reason = True, "all text is hidden"
        else:
            hide, reason = decide(line["text"], fingerprint, modality, common_codes, line["confidence"])
        decisions.append({"box": line["box"], "text": line["text"], "confidence": round(line["confidence"], 3),
                          "hide": hide, "reason": reason})
        if hide:
            boxes.append(line["box"])
    return boxes, decisions


def common_codes_from(ocr_by_file, patient_of_file, min_patients=3):
    """Short codes that appear in the images of many different patients are UI/device words, not initials."""
    patients_per_code = {}
    for path, found in ocr_by_file.items():
        for box, text, confidence in found:
            code = text.strip().upper()
            if INITIALS.match(code):
                patients_per_code.setdefault(code, set()).add(patient_of_file[path])
    return set(code for code, patients in patients_per_code.items() if len(patients) >= min_patients)


# Apply to the output files of the engine

ALWAYS_READ = ["CR", "DX", "MG", "US", "SC", "OT", "XA", "RF", "IO", "PX", "DG", "ES", "XC"]


def load_pixels(path):
    ds = pydicom.dcmread(path)
    # some colour images miss Planar Configuration; 0 (colour-by-pixel) is the usual default
    if int(ds.get("SamplesPerPixel", 1) or 1) > 1 and "PlanarConfiguration" not in ds:
        ds.PlanarConfiguration = 0
    array = ds.pixel_array.copy()
    photometric = str(ds.get("PhotometricInterpretation", ""))
    if photometric.startswith("YBR"):
        from pydicom.pixels import convert_color_space
        array = convert_color_space(array, photometric, "RGB")
        photometric = "RGB"
    return ds, array, photometric


def first_frame(ds, array):
    if int(ds.get("NumberOfFrames", 1) or 1) > 1:
        return array[0]
    return array


def has_pixels(path):
    """Image files have a size (Rows); structured reports and other non-image files do not.
    (The header is read without the pixel data, so we cannot look for Pixel Data itself.)"""
    ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["Rows"])
    return "Rows" in ds


def ocr_file(path):
    try:
        ds, array, photometric = load_pixels(path)
    except Exception:
        return []                       # pixels that cannot be decoded are left as they are (listed in the audit)
    return ocr_image(first_frame(ds, array))


def has_phi_text(found, fingerprint, modality):
    """Screening decision: does this OCR result contain at least one line that would be hidden?"""
    for line in group_lines(found):
        hide, reason = decide(line["text"], fingerprint, modality)
        if hide:
            return True
    return False


def screening_plan(records, modality_of, series_of):
    """Projection images are always read; for CT/MR/PET series only the first, middle and last slice."""
    always = []
    by_series = {}
    for record in records:
        if record["output"] == "":
            continue
        if not has_pixels(record["output"]):
            continue                    # e.g. structured reports: no image, nothing to read
        if modality_of[record["input"]] in ALWAYS_READ:
            always.append(record)
        else:
            by_series.setdefault(series_of[record["input"]], []).append(record)
    samples = {}
    for series, items in by_series.items():
        items = sorted(items, key=lambda r: r["input"])
        picks = []
        for pick in [items[0], items[len(items) // 2], items[-1]]:
            if pick not in picks:
                picks.append(pick)
        samples[series] = (picks, items)
    return always, samples


class OcrCache:
    """OCR results saved line by line (output path -> text found), so a stopped run can go on where it was."""

    def __init__(self, cache_path=None):
        self.cache_path = cache_path
        self.saved = {}
        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue          # the last line of a stopped run can be cut in half
                    self.saved[entry["output"]] = [tuple(item) for item in entry["found"]]

    def read(self, output_path):
        if output_path in self.saved:
            return self.saved[output_path]
        found = ocr_file(output_path)
        if self.cache_path is not None:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"output": output_path, "found": found}, default=float) + "\n")
        self.saved[output_path] = found
        return found


def clean_outputs(records, fingerprints, patient_of, modality_of, series_of, audit_path, method="fpdeid",
                  ocr_cache_path=None):
    """Pass 1: OCR (with series screening). Pass 2: decide and hide. One audit line per file that was read.

    With ocr_cache_path, OCR results are saved as they are made and reused when the run is started again.
    """
    start = time.time()
    cache = OcrCache(ocr_cache_path)
    always, samples = screening_plan(records, modality_of, series_of)
    ocr_by_file = {}
    skipped = 0
    for record in always:
        ocr_by_file[record["output"]] = cache.read(record["output"])
    for series, (picks, items) in samples.items():
        any_text = False
        for record in picks:
            ocr_by_file[record["output"]] = cache.read(record["output"])
            fingerprint = fingerprints.get(patient_of[record["input"]])
            if has_phi_text(ocr_by_file[record["output"]], fingerprint, modality_of[record["input"]]):
                any_text = True
        rest = [r for r in items if r not in picks]
        if any_text:
            for record in rest:
                ocr_by_file[record["output"]] = cache.read(record["output"])
        else:
            skipped = skipped + len(rest)
    ocr_seconds = time.time() - start

    input_of = {r["output"]: r["input"] for r in records}
    patient_of_output = {out: patient_of[inp] for out, inp in input_of.items()}
    common_codes = common_codes_from(ocr_by_file, patient_of_output)

    hidden_files = 0
    failed = 0
    for output_path, found in ocr_by_file.items():
        input_path = input_of[output_path]
        fingerprint = fingerprints.get(patient_of[input_path])
        try:
            ds, array, photometric = load_pixels(output_path)
        except Exception as problem:
            failed = failed + 1
            write_audit(audit_path, input_path, output_path, [], [{"error": str(problem)[:200]}])
            continue
        frame = first_frame(ds, array)
        boxes, decisions = boxes_for_method(found, method, fingerprint, modality_of[input_path], common_codes,
                                            frame.shape[0], frame.shape[1])
        if len(boxes) > 0:
            if int(ds.get("NumberOfFrames", 1) or 1) > 1:
                for i in range(array.shape[0]):
                    array[i] = hide_boxes(array[i], boxes, photometric)
            else:
                array = hide_boxes(array, boxes, photometric)
            ds.PixelData = array.tobytes()
            ds.PhotometricInterpretation = photometric
            if photometric == "RGB":
                ds.PlanarConfiguration = 0
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.BurnedInAnnotation = "NO"
            ds.save_as(output_path, enforce_file_format=False)
            hidden_files = hidden_files + 1
        write_audit(audit_path, input_path, output_path, boxes, decisions)

    return {
        "files read by OCR": len(ocr_by_file),
        "files skipped by series screening": skipped,
        "files with hidden text": hidden_files,
        "files whose pixels could not be read": failed,
        "common codes (kept)": len(common_codes),
        "OCR seconds": round(ocr_seconds, 1),
        "total seconds": round(time.time() - start, 1),
    }


def write_audit(audit_path, input_path, output_path, boxes, decisions):
    entry = {"input": input_path, "output": output_path, "hidden_boxes": len(boxes), "boxes": boxes,
             "decisions": decisions, "ps315_option": "clean_pixel_data",
             "gdpr": "Data minimisation (Art. 5(1)(c)): identifying text in the image hidden, the rest of the image kept"}
    with open(audit_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
