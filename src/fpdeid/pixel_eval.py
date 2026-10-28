"""Phase 4: score pixel-redaction methods on the synthetic burned-in text benchmark."""

import re

import numpy as np

from fpdeid import pixels


def covered_share(box, hidden_boxes):
    """Share of `box` that lies inside the union of hidden boxes (with the same margin as hide_boxes)."""
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    mask = np.zeros((height, width), dtype=bool)
    for hx0, hy0, hx1, hy1 in hidden_boxes:
        margin = max(4, int(0.2 * (hy1 - hy0)))
        hx0, hy0, hx1, hy1 = hx0 - margin, hy0 - margin, hx1 + margin, hy1 + margin
        ax0 = max(x0, hx0) - x0
        ay0 = max(y0, hy0) - y0
        ax1 = min(x1, hx1) - x0
        ay1 = min(y1, hy1) - y0
        if ax1 > ax0 and ay1 > ay0:
            mask[ay0:ay1, ax0:ax1] = True
    return mask.mean()


def overlaps(box, other):
    return not (box[2] <= other[0] or box[0] >= other[2] or box[3] <= other[1] or box[1] >= other[3])


def decisions_for_method(found, method, fingerprint, height, width, modality="", common_codes=None,
                         min_confidence=0.0):
    """found = OCR result [(box, text, conf)] -> list of boxes to hide for this method."""
    boxes, decisions = pixels.boxes_for_method(found, method, fingerprint, modality, common_codes, height, width,
                                               min_confidence)
    return boxes


def words_of(text):
    return [w for w in re.findall(r"[A-Za-z0-9]+", text.upper()) if len(w) >= 3]


def score_file(labels, hidden_boxes, height, width):
    phi = [label for label in labels if label["phi"]]
    harmless = [label for label in labels if not label["phi"]]
    result = {}
    result["phi_boxes"] = len(phi)
    result["phi_hidden"] = sum(1 for label in phi if covered_share(label["box"], hidden_boxes) >= 0.9)
    result["harmless_boxes"] = len(harmless)
    result["harmless_kept"] = sum(1 for label in harmless if covered_share(label["box"], hidden_boxes) < 0.1)
    result["hidden_boxes"] = len(hidden_boxes)
    result["hidden_on_phi"] = sum(1 for box in hidden_boxes if any(overlaps(box, label["box"]) for label in phi))
    area = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in hidden_boxes:
        pad = max(4, int(0.2 * (y1 - y0)))
        area[max(0, y0 - pad):min(height, y1 + pad), max(0, x0 - pad):min(width, x1 + pad)] = True
    result["hidden_area_pct"] = 100 * area.mean()
    return result


def leaked_words(image8, labels):
    """OCR the PHI boxes of the image AFTER hiding. Returns the number of PHI words that are still readable."""
    leaked = 0
    height, width = image8.shape[0], image8.shape[1]
    for label in labels:
        if not label["phi"]:
            continue
        x0, y0, x1, y1 = label["box"]
        pad = 10
        crop = image8[max(0, y0 - pad):min(height, y1 + pad), max(0, x0 - pad):min(width, x1 + pad)]
        if crop.size == 0:
            continue
        read = " ".join(text for box, text, confidence in pixels.read_text(crop, min_confidence=0.1))
        read_words = set(words_of(read))
        for word in words_of(label["text"]):
            if word in read_words:
                leaked = leaked + 1
    return leaked
