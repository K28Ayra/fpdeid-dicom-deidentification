"""Where does a patient's PHI appear OUTSIDE its home tag?

For every file we take the values of a few "home" tags (name, ID, birth date, accession number)
and look for them inside every other text value of the same file (also private tags and sequences).
If they are found elsewhere, a tag-by-tag rule system that keeps that other tag would leak them.
This is the motivation for the PHI fingerprint (Phase 3).
"""

from multiprocessing import Pool

import pandas as pd
import pydicom

from fpdeid.census import tag_label

HOME_TAGS = ["PatientName", "PatientID", "PatientBirthDate", "AccessionNumber", "OtherPatientIDs"]
TEXT_VRS = ["LO", "LT", "SH", "ST", "UT", "PN", "CS", "UC", "AE", "DA", "DT", "UN", "OB"]


def home_values(ds):
    """Return [(home_tag, text_to_look_for)]. Names are split into parts (DOE^JOHN -> DOE, JOHN)."""
    values = []
    for keyword in HOME_TAGS:
        if keyword not in ds:
            continue
        raw = str(ds.data_element(keyword).value).strip()
        if raw == "":
            continue
        if keyword == "PatientName":
            parts = raw.replace("^", " ").split()
            for part in parts:
                if len(part) >= 3:
                    values.append((keyword, part.upper()))
        else:
            if len(raw) >= 4:
                values.append((keyword, raw.upper()))
    return values


def element_text(element):
    if element.VR in ["OB", "UN"]:
        try:
            return bytes(element.value).decode("latin-1", errors="ignore").upper()
        except Exception:
            return ""
    return str(element.value).upper()


def search_file(file_path):
    hits = []
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
    except Exception:
        return hits
    targets = home_values(ds)
    if len(targets) == 0:
        return hits
    modality = str(ds.get("Modality", ""))

    for element in ds.iterall():
        if element.VR not in TEXT_VRS:
            continue
        if element.keyword in HOME_TAGS:
            continue
        if element.keyword == "PixelData":
            continue
        text = element_text(element)
        if text == "":
            continue
        for home_tag, value in targets:
            if value in text:
                hit = {
                    "file": file_path,
                    "modality": modality,
                    "home_tag": home_tag,
                    "found_in": tag_label(element),
                    "found_in_name": element.name,
                    "private": element.tag.is_private,
                }
                hits.append(hit)
    return hits


def find_phi_outside_home(file_paths, workers=16):
    with Pool(workers) as pool:
        per_file = pool.map(search_file, file_paths, chunksize=50)
    rows = []
    for hits in per_file:
        rows.extend(hits)
    return pd.DataFrame(rows)
