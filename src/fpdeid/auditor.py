"""Phase 5: an INDEPENDENT auditor of de-identified files.

It does not import any code of the cleaner (engine, fingerprint, pixels). It has its own, simple idea of
"who is this patient":

1. From the ORIGINAL files of a patient it collects the identity: every name (all PN fields), the IDs,
   the accession numbers, the original dates and times of birth/study, addresses, phone numbers,
   institution and station names, device serial numbers, and every original UID.
2. In every OUTPUT file it reads every value (also inside sequences and private tags, also bytes that
   are readable text) and looks for that identity, in several written forms.
3. On images where text is likely to be written into the pixels it runs its own OCR and does the same.

The result is a list of hits: file, tag, what kind of identifier was found, and the matched text.
"""

import json
import re
from multiprocessing import Pool

import numpy as np
import pydicom

WORKER = {}

# fields whose values identify the patient, the visit or the site (besides all PN = person names)
ID_KEYWORDS = ["PatientID", "OtherPatientIDs", "AccessionNumber", "StudyID", "MedicalRecordLocator",
               "IssuerOfPatientID", "AdmissionID", "PerformedProcedureStepID", "RequestedProcedureID",
               "ScheduledProcedureStepID", "FillerOrderNumberImagingServiceRequest",
               "PlacerOrderNumberImagingServiceRequest", "DeviceSerialNumber", "PatientInsurancePlanCodeSequence",
               "ClinicalTrialSubjectID", "ClinicalTrialSubjectReadingID"]
PLACE_KEYWORDS = ["InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName", "StationName",
                  "PatientAddress", "RegionOfResidence", "CountryOfResidence", "PatientTelephoneNumbers",
                  "MilitaryRank", "BranchOfService", "PatientBirthName", "PatientMotherBirthName"]
DATE_KEYWORDS = ["PatientBirthDate", "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# words that are never a name on their own (titles, sex, very short pieces)
NOT_NAME_WORDS = set(["MR", "MRS", "MS", "DR", "MD", "PHD", "JR", "SR", "II", "III", "THE", "AND", "VAN", "VON",
                      "DE", "DER", "LA", "LE", "DEL", "MALE", "FEMALE", "UNKNOWN", "ANONYMOUS", "NONE", "PATIENT"])

# placeholder values that are written in many files and identify nobody
PLACEHOLDERS = set(["NONE", "UNKNOWN", "ANONYMOUS", "N/A", "NA", "NULL", "-", "0000", "00000000", "REMOVED"])

# coded values come from a fixed dictionary (e.g. "Long axis"), they cannot hold a patient's identity
CODED_KEYWORDS = set(["CodeValue", "CodeMeaning", "CodingSchemeDesignator", "CodingSchemeVersion", "LongCodeValue",
                      "URNCodeValue", "MappingResource", "ContextIdentifier", "ContextUID"])

TEXT_VRS = ["AE", "AS", "CS", "DA", "DS", "DT", "IS", "LO", "LT", "PN", "SH", "ST", "TM", "UC", "UR", "UT", "UN",
            "OB"]


# ---------------------------------------------------------------
# 1. The identity of a patient, from the original files
# ---------------------------------------------------------------

def element_texts(element):
    """All values of an element as a list of strings (bytes are decoded if they are readable)."""
    value = element.value
    if value is None:
        return []
    if isinstance(value, bytes):
        text = value.decode("latin-1", errors="ignore")
        printable = sum(1 for c in text if c.isprintable())
        if len(text) == 0 or printable < 0.8 * len(text):
            return []
        return [text]
    if element.VM > 1:
        return [str(v) for v in value]
    return [str(value)]


def date_forms(date8):
    """'20170613' -> the common ways to write this date."""
    if len(date8) != 8 or not date8.isdigit():
        return []
    year = date8[0:4]
    month = date8[4:6]
    day = date8[6:8]
    forms = [date8,
             year + "-" + month + "-" + day,
             year + "/" + month + "/" + day,
             year + "." + month + "." + day,
             month + "/" + day + "/" + year,
             day + "/" + month + "/" + year,
             day + "." + month + "." + year,
             month + "-" + day + "-" + year,
             day + "-" + month + "-" + year]
    # with the month as a word: 13 JUN 2017, 13-JUN-2017, JUN 13 2017, JUN 13, 2017
    if "01" <= month <= "12":
        word = MONTHS[int(month) - 1]
        forms.append(day + " " + word + " " + year)
        forms.append(day + "-" + word + "-" + year)
        forms.append(word + " " + day + " " + year)
        forms.append(word + " " + day + ", " + year)
    return forms


def is_private_creator(element):
    """(gggg,00xx) in an odd group: the name of a vendor's private block, not a value."""
    return element.tag.group % 2 == 1 and 0x0010 <= element.tag.element <= 0x00FF


def is_class_uid(element):
    """UIDs that name a TYPE of object (SOP class, transfer syntax), the same for every patient."""
    keyword = element.keyword
    if "ClassUID" in keyword or "TransferSyntax" in keyword or "SOPClasses" in keyword:
        return True
    if element.tag == 0x00020012:          # Implementation Class UID
        return True
    return False


def add_identity_value(identity, kind, text):
    text = text.strip().upper()
    if text == "" or text in PLACEHOLDERS:
        return
    identity.setdefault(kind, set()).add(text)


def collect_identity(ds, identity):
    """Add the identifying values of one original file to the identity dict {kind: set of strings}."""
    for element in ds.iterall():
        keyword = element.keyword
        if is_private_creator(element):
            continue
        if element.keyword in CODED_KEYWORDS:
            continue
        if element.VR == "PN":
            for text in element_texts(element):
                add_identity_value(identity, "full name", text.replace("^", " "))
                for word in re.split(r"[\^ ,.]+", text):
                    word = word.strip().upper()
                    if len(word) >= 3 and word not in NOT_NAME_WORDS:
                        add_identity_value(identity, "name word", word)
        elif keyword in ID_KEYWORDS:
            for text in element_texts(element):
                if len(text.strip()) >= 4:
                    add_identity_value(identity, "id", text)
        elif keyword in PLACE_KEYWORDS:
            for text in element_texts(element):
                if len(text.strip()) >= 4:
                    add_identity_value(identity, "place or contact", text)
        elif keyword in DATE_KEYWORDS:
            for text in element_texts(element):
                for form in date_forms(text.strip()[:8]):
                    add_identity_value(identity, "date", form)
        elif element.VR == "UI" and not is_class_uid(element):
            for text in element_texts(element):
                if not text.startswith("1.2.840.10008."):
                    add_identity_value(identity, "uid", text)
    return identity


def identity_of_files(file_paths):
    identity = {}
    for path in file_paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        collect_identity(ds, identity)
    return identity


def build_matchers(identity):
    """One regular expression per kind. Words must stand alone (not inside a longer word or number)."""
    matchers = {}
    for kind, values in identity.items():
        if kind == "uid":
            continue          # UIDs are compared as whole values, not searched inside text
        pieces = []
        for value in sorted(values, key=len, reverse=True):
            pieces.append(re.escape(value))
        if len(pieces) == 0:
            continue
        pattern = r"(?<![A-Z0-9])(" + "|".join(pieces) + r")(?![A-Z0-9])"
        matchers[kind] = re.compile(pattern)
    return matchers


# ---------------------------------------------------------------
# 2. Scan the text of an output file
# ---------------------------------------------------------------

def find_in_text(text, matchers):
    """[(kind, matched text)] for one piece of text."""
    found = []
    upper = text.upper()
    for kind, matcher in matchers.items():
        for match in matcher.finditer(upper):
            found.append((kind, match.group(1)))
    return found


def walk(ds, parent, rows):
    """Every element as (tag path, VR, element), also inside sequences."""
    for element in ds:
        tag = "(%04X,%04X)" % (element.tag.group, element.tag.element)
        path = parent + tag
        if element.VR == "SQ":
            for item in element.value:
                walk(item, path + ">", rows)
        else:
            rows.append((path, element.VR, element))
    return rows


def scan_header(output_path, matchers, original_uids):
    """Returns (hits, number of elements read)."""
    ds = pydicom.dcmread(output_path, stop_before_pixels=True)
    hits = []
    elements = walk(ds, "", [])
    file_meta_uid = str(ds.file_meta.get("MediaStorageSOPInstanceUID", ""))
    if file_meta_uid in original_uids:
        hits.append({"file": output_path, "tag": "(0002,0003)", "kind": "uid", "text": file_meta_uid})
    for path, vr, element in elements:
        if element.keyword == "PixelData" or is_private_creator(element) or element.keyword in CODED_KEYWORDS:
            continue
        if vr == "UI":
            for text in element_texts(element):
                if text in original_uids:
                    hits.append({"file": output_path, "tag": path, "kind": "uid", "text": text})
            continue
        if vr not in TEXT_VRS:
            continue
        for text in element_texts(element):
            for kind, matched in find_in_text(text, matchers):
                hits.append({"file": output_path, "tag": path, "kind": kind, "text": matched})
    return hits, len(elements)


# ---------------------------------------------------------------
# 3. Text in the pixels (its own OCR)
# ---------------------------------------------------------------

def get_reader():
    if "reader" not in WORKER:
        import easyocr
        WORKER["reader"] = easyocr.Reader(["en"], gpu=True, verbose=False)
    return WORKER["reader"]


def pixel_views(array):
    """Two 8-bit views: the full value range, and only the brightest values (where burned-in text usually is)."""
    image = array.astype(np.float32)
    if image.ndim == 3:
        image = image.mean(axis=2)
    low = image.min()
    high = image.max()
    if high <= low:
        return []
    full = (255 * (image - low) / (high - low)).astype(np.uint8)
    top = np.percentile(image, 99.5)
    bright = np.clip(255 * (image - top) / max(1e-6, high - top), 0, 255).astype(np.uint8)
    views = [full, bright]
    # small images are enlarged, OCR cannot read 6-pixel letters
    longest = max(full.shape[0], full.shape[1])
    if longest < 1000:
        import cv2
        scale = 1000 / longest
        resized = []
        for view in views:
            size = (int(view.shape[1] * scale), int(view.shape[0] * scale))
            resized.append(cv2.resize(view, size, interpolation=cv2.INTER_CUBIC))
        views = resized
    return views


def read_pixel_text(output_path):
    """All text the OCR reads in the first frame of an output image (one string per piece)."""
    ds = pydicom.dcmread(output_path)
    if "PixelData" not in ds:
        return []
    array = ds.pixel_array
    frames = int(ds.get("NumberOfFrames", 1) or 1)
    if frames > 1:
        array = array[0]
    if array.ndim == 3 and ds.get("SamplesPerPixel", 1) == 3 and str(ds.get("PhotometricInterpretation", "")).startswith("YBR"):
        array = pydicom.pixels.convert_color_space(array, str(ds.PhotometricInterpretation), "RGB")
    reader = get_reader()
    texts = []
    for view in pixel_views(array):
        for box, text, confidence in reader.readtext(view):
            if confidence >= 0.1:
                texts.append(text)
    return texts


def fuzzy_word_hits(texts, identity):
    """OCR makes small mistakes: a name word or ID of 5+ characters also counts if it is read 85% the same."""
    from rapidfuzz import fuzz
    hits = []
    candidates = []
    for kind in ["name word", "id"]:
        for value in identity.get(kind, set()):
            if len(value) >= 5:
                candidates.append((kind, value))
    for text in texts:
        for word in re.findall(r"[A-Z0-9]+", text.upper()):
            if len(word) < 5:
                continue
            for kind, value in candidates:
                if fuzz.ratio(word, value) >= 85:
                    hits.append((kind + " (OCR, fuzzy)", word))
    return hits


def scan_pixels(output_path, matchers, identity):
    texts = read_pixel_text(output_path)
    hits = []
    joined = " ".join(texts)
    for kind, matched in find_in_text(joined, matchers):
        hits.append({"file": output_path, "tag": "pixels", "kind": kind, "text": matched})
    for kind, matched in fuzzy_word_hits(texts, identity):
        hits.append({"file": output_path, "tag": "pixels", "kind": kind, "text": matched})
    return hits, texts


# ---------------------------------------------------------------
# 4. A whole run: many patients, many files (header part in parallel, CPU only)
# ---------------------------------------------------------------

def audit_patient(job):
    """job = (original file paths, output file paths) of one patient -> (hits, elements read)."""
    original_paths, output_paths = job
    identity = identity_of_files(original_paths)
    matchers = build_matchers(identity)
    original_uids = identity.get("uid", set())
    hits = []
    elements = 0
    for output_path in output_paths:
        file_hits, count = scan_header(output_path, matchers, original_uids)
        hits.extend(file_hits)
        elements = elements + count
    return hits, elements


def audit_headers(jobs, workers=16):
    with Pool(workers) as pool:
        results = pool.map(audit_patient, jobs, chunksize=1)
    hits = []
    elements = 0
    for patient_hits, count in results:
        hits.extend(patient_hits)
        elements = elements + count
    return hits, elements


def audit_pixel_files(jobs):
    """jobs = [(original file paths of the patient, output image path)] -> (hits, OCR text per file). GPU."""
    hits = []
    texts_by_file = {}
    identity_cache = {}
    for original_paths, output_path in jobs:
        key = original_paths[0]
        if key not in identity_cache:
            identity = identity_of_files(original_paths)
            identity_cache[key] = (identity, build_matchers(identity))
        identity, matchers = identity_cache[key]
        try:
            file_hits, texts = scan_pixels(output_path, matchers, identity)
        except Exception as problem:
            file_hits = []
            texts = ["<could not read: " + str(problem)[:100] + ">"]
        hits.extend(file_hits)
        texts_by_file[output_path] = texts
    return hits, texts_by_file


# ---------------------------------------------------------------
# 5. Closed loop: fix what the auditor found (header hits), and log every fix
# ---------------------------------------------------------------

def elements_at(ds, tag_path):
    """All elements at a tag path like '(0040,A730)>(0040,A160)' (every item of the sequences)."""
    parts = tag_path.split(">")
    current = [ds]
    for i, part in enumerate(parts):
        group = int(part[1:5], 16)
        number = int(part[6:10], 16)
        tag = (group << 16) + number
        found = []
        for dataset in current:
            if tag in dataset:
                found.append(dataset[tag])
        if i == len(parts) - 1:
            return found
        current = []
        for element in found:
            if element.VR == "SQ":
                current.extend(list(element.value))
    return []


def remove_text(value, matched):
    pattern = re.compile(re.escape(matched), re.IGNORECASE)
    return pattern.sub("", value).strip()


def fix_element(element, hit):
    if hit["kind"] == "uid":
        from pydicom.uid import generate_uid
        element.value = generate_uid(prefix="2.25.", entropy_srcs=[hit["text"]])
        return
    value = element.value
    if isinstance(value, bytes):
        text = value.decode("latin-1", errors="ignore")
        element.value = remove_text(text, hit["text"]).encode("latin-1", errors="ignore")
    elif element.VM > 1:
        element.value = [remove_text(str(v), hit["text"]) for v in value]
    else:
        element.value = remove_text(str(value), hit["text"])


def fix_header_hits(hits, log_path):
    """Remove every matched identifier from the output files (in place). Returns the number of files changed."""
    by_file = {}
    for hit in hits:
        if hit["tag"] == "pixels":
            continue                       # pixel hits are listed for a human to look at
        by_file.setdefault(hit["file"], []).append(hit)
    with open(log_path, "a") as log:
        for path, file_hits in by_file.items():
            ds = pydicom.dcmread(path)
            for hit in file_hits:
                if hit["tag"] == "(0002,0003)":
                    continue                   # follows the SOP Instance UID below
                for element in elements_at(ds, hit["tag"]):
                    fix_element(element, hit)
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            ds.save_as(path, enforce_file_format=False)
            for hit in file_hits:
                entry = {"file": path, "tag": hit["tag"], "kind": hit["kind"], "action": "removed by auditor",
                         "ps315_option": "basic_profile", "gdpr": "Art. 4(5) pseudonymisation: residual identifier"}
                log.write(json.dumps(entry) + "\n")
    return len(by_file)
