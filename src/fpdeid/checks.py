"""Independent checks of a de-identification run (Phase 2 evaluation).

They do NOT use the engine's code: they only read the input file, the output file and the audit log,
and compare them. One function (check_pair) reads both headers once and measures:

- audit coverage   : every element that changed / disappeared / appeared has an audit entry
- utility          : research-critical fields kept with the same value
- date intervals   : the gaps between dates of the same file are the same before and after
- UID safety       : no original UID is left anywhere, and all UIDs are valid
"""

import datetime
import re
from multiprocessing import Pool

import pydicom
import yaml

from fpdeid import paths

WORKER = {}
DATE_KEYWORDS = ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"]
UID_PATTERN = re.compile(r"^[0-9]+(\.[0-9]+)*$")


def load_utility_keywords():
    with open(paths.CONFIGS + "/utility_tags.yaml") as f:
        groups = yaml.safe_load(f)
    keywords = []
    for group in groups.values():
        keywords.extend(group)
    return keywords


def flatten(ds, parent, table):
    """{path: [values]} for every element (item numbers are left out of the path, like in the audit log)."""
    for element in ds:
        tag = "(%04X,%04X)" % (element.tag.group, element.tag.element)
        path = parent + tag
        if element.VR == "SQ":
            table.setdefault(path, []).append("<sequence of %d>" % len(element.value))
            for item in element.value:
                flatten(item, path + ">", table)
        elif element.keyword == "PixelData":
            continue
        else:
            table.setdefault(path, []).append(str(element.value))
    return table


def all_uids(ds, found):
    for element in ds.iterall():
        if element.VR == "UI" and element.value:
            if element.VM > 1:
                for value in element.value:
                    found.append(str(value))
            else:
                found.append(str(element.value))
    return found


def to_date(text):
    try:
        return datetime.datetime.strptime(str(text)[:8], "%Y%m%d")
    except ValueError:
        return None


def start_worker(input_uids):
    WORKER["input_uids"] = input_uids
    WORKER["utility"] = load_utility_keywords()


def check_pair(job):
    input_path, output_path, audited_paths = job
    result = {"input": input_path}

    original = pydicom.dcmread(input_path, stop_before_pixels=True)
    cleaned = pydicom.dcmread(output_path, stop_before_pixels=True)

    # --- audit coverage ---
    before = flatten(original, "", {})
    after = flatten(cleaned, "", {})
    changed = []
    for path in set(before) | set(after):
        if before.get(path) != after.get(path):
            changed.append(path)
    not_logged = []
    for path in changed:
        if path not in audited_paths and not is_inside_logged_sequence(path, audited_paths):
            not_logged.append(path)
    result["changed_paths"] = len(changed)
    result["changed_not_logged"] = len(not_logged)
    result["not_logged_examples"] = " | ".join(sorted(not_logged)[:3])

    # --- utility: research-critical fields ---
    present = 0
    kept = 0
    lost = []
    for keyword in WORKER["utility"]:
        if keyword in original:
            present = present + 1
            if keyword in cleaned and str(cleaned.data_element(keyword).value) == str(original.data_element(keyword).value):
                kept = kept + 1
            else:
                lost.append(keyword)
    result["utility_present"] = present
    result["utility_kept"] = kept
    result["utility_lost"] = ",".join(lost)

    # --- date intervals inside the file ---
    pairs = 0
    same = 0
    for first in DATE_KEYWORDS:
        for second in DATE_KEYWORDS:
            if first >= second:
                continue
            a0 = to_date(original.get(first, ""))
            b0 = to_date(original.get(second, ""))
            a1 = to_date(cleaned.get(first, ""))
            b1 = to_date(cleaned.get(second, ""))
            if a0 is None or b0 is None:
                continue
            pairs = pairs + 1
            if a1 is not None and b1 is not None and (b1 - a1) == (b0 - a0):
                same = same + 1
    result["date_pairs"] = pairs
    result["date_pairs_same"] = same

    # --- UIDs ---
    output_uids = all_uids(cleaned, [])
    output_uids.append(str(cleaned.file_meta.MediaStorageSOPInstanceUID))
    left_over = 0
    invalid = 0
    for uid in output_uids:
        if uid in WORKER["input_uids"] and not uid.startswith("1.2.840.10008."):
            left_over = left_over + 1
        if len(uid) > 64 or not UID_PATTERN.match(uid):
            invalid = invalid + 1
    result["original_uids_left"] = left_over
    result["invalid_uids"] = invalid
    return result


def is_inside_logged_sequence(path, audited_paths):
    """A change inside a sequence that was removed or emptied as a whole is covered by that entry."""
    parts = path.split(">")
    for i in range(1, len(parts)):
        parent = ">".join(parts[:i])
        if parent in audited_paths:
            return True
    return False


def run_checks(jobs, input_uids, workers=16):
    with Pool(workers, initializer=start_worker, initargs=(input_uids,)) as pool:
        rows = pool.map(check_pair, jobs, chunksize=20)
    return rows


# ---------------------------------------------------------------
# Residual fingerprint hits: is any of the patient's identity still in the output?
# ---------------------------------------------------------------

TEXT_LIKE_VRS = ["LO", "SH", "ST", "LT", "UT", "UC", "PN", "CS", "AE", "UN", "OB"]


def start_residual_worker(fingerprints):
    WORKER["fingerprints"] = fingerprints


def residual_hits(job):
    """job = (output_path, original_patient_id). Returns a list of (tag path, category, source) still found."""
    output_path, patient_id = job
    fingerprint = WORKER["fingerprints"].get(patient_id)
    if fingerprint is None:
        return []
    ds = pydicom.dcmread(output_path, stop_before_pixels=True)
    hits = []
    for element in ds.iterall():
        if element.VR not in TEXT_LIKE_VRS or element.value is None:
            continue
        if element.VR in ["UN", "OB"]:
            try:
                text = bytes(element.value).decode("latin-1", errors="ignore")
            except Exception:
                continue
        else:
            text = str(element.value)
        if text.strip() == "":
            continue
        for start, end, category, source in fingerprint.find(text):
            if source == "generic":
                continue          # only the patient's own identity counts here
            hits.append(("(%04X,%04X)" % (element.tag.group, element.tag.element), category, source,
                         output_path))
    return hits


def run_residual_check(jobs, fingerprints, workers=16):
    with Pool(workers, initializer=start_residual_worker, initargs=(fingerprints,)) as pool:
        per_file = pool.map(residual_hits, jobs, chunksize=20)
    rows = []
    for hits in per_file:
        rows.extend(hits)
    return rows
