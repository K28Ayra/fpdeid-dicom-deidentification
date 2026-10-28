"""Baseline de-identifiers for Phase 1."""

import datetime
import hashlib
import os
from multiprocessing import Pool

import pandas as pd
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from fpdeid import ps315

BASELINE_OPTIONS = {
    "B0": None,
    "BA": [],
    "BB": ["retain_device_id", "retain_patient_chars", "retain_long_modified_dates"],
}

# how the method is recorded in the file (PS3.15 / PS3.16 CID 7050 codes)
METHOD_CODES = {
    "B0": [],
    "BA": [("113100", "Basic Application Confidentiality Profile")],
    "BB": [
        ("113100", "Basic Application Confidentiality Profile"),
        ("113107", "Retain Longitudinal Temporal Information Modified Dates Option"),
        ("113108", "Retain Patient Characteristics Option"),
        ("113109", "Retain Device Identity Option"),
    ],
}


# Pseudonyms: UIDs, patient IDs, date shift

class Mappings:
    """Holds the old->new tables. Unknown UIDs get a stable hash-based UID."""

    def __init__(self, uid_map_csv, patient_map_csv, secret=None):
        # the tables are optional: without them every pseudonym comes from the secret (hash)
        self.uids = {}
        self.patients = {}
        if uid_map_csv is not None:
            uid_table = pd.read_csv(uid_map_csv, dtype=str)
            self.uids = dict(zip(uid_table["id_old"], uid_table["id_new"]))
        if patient_map_csv is not None:
            patient_table = pd.read_csv(patient_map_csv, dtype=str)
            self.patients = dict(zip(patient_table["id_old"], patient_table["id_new"]))
        # the secret is the pseudonymisation key: keep it with the data custodian (FPDEID_SECRET)
        if secret is None:
            secret = os.environ.get("FPDEID_SECRET", "ayra-project2")
        self.secret = secret

    def new_uid(self, old_uid):
        old_uid = str(old_uid).strip()
        if old_uid in self.uids:
            return self.uids[old_uid]
        # not in the table: make a new UID from a hash (same input -> same output)
        text = self.secret + old_uid
        digest = hashlib.sha256(text.encode()).hexdigest()
        number = int(digest[:30], 16)
        return "2.25." + str(number)

    def new_patient_id(self, old_id):
        old_id = str(old_id).strip()
        if old_id in self.patients:
            return self.patients[old_id]
        text = self.secret + old_id
        digest = hashlib.sha256(text.encode()).hexdigest()
        return "PSEUDO_" + digest[:10].upper()

    def date_offset_days(self, old_patient_id):
        """Same patient -> same shift, so time intervals are kept. Shift is 1 to 10 years back."""
        text = self.secret + "date" + str(old_patient_id)
        digest = hashlib.sha256(text.encode()).hexdigest()
        number = int(digest[:8], 16)
        days = 365 + (number % 3285)
        return -days


def shift_date_text(value, offset_days):
    """'20120311' -> shifted date. Returns '' if the date cannot be read."""
    value = str(value).strip()
    if len(value) < 8:
        return ""
    try:
        date = datetime.datetime.strptime(value[:8], "%Y%m%d")
    except ValueError:
        return ""
    new_date = date + datetime.timedelta(days=offset_days)
    return new_date.strftime("%Y%m%d") + value[8:]


# Actions on one element

def dummy_value(element, mappings, patient_id):
    vr = element.VR
    if element.keyword == "PatientID":
        return mappings.new_patient_id(patient_id)
    if element.keyword == "PatientName":
        return mappings.new_patient_id(patient_id)
    if vr == "DA":
        return "19000101"
    if vr == "TM":
        return "000000"
    if vr == "DT":
        return "19000101000000"
    if vr in ["IS", "DS", "US", "SS", "UL", "SL", "FL", "FD"]:
        return 0
    if vr == "AS":
        return "000Y"
    if vr == "UI":
        return mappings.new_uid(element.value)
    return "ANONYMIZED"


def apply_action(ds, element, action, mappings, patient_id, offset_days):
    """Change one element in place. Returns the action really done (for counting)."""
    tag = element.tag

    # For a sequence, "U" (new UIDs), "C" (clean) or "D" (dummy) means: work on the elements INSIDE it.
    # Returning "K" makes deid_dataset go into the sequence items.
    # For "D" (dummy) on a sequence we also keep the items and clean their contents:
    # an empty dummy item would miss attributes that the sequence requires.
    if element.VR == "SQ" and action in ["U", "C", "D"]:
        return "K"

    if action == "X":
        del ds[tag]
        return "X"

    if action == "Z":
        if element.VR == "SQ":
            element.value = Sequence([])
        else:
            element.value = ""
        return "Z"

    if action == "D":
        if element.VR == "SQ":
            element.value = Sequence([Dataset()])
        else:
            element.value = dummy_value(element, mappings, patient_id)
        return "D"

    if action == "U":
        if element.value is None or element.value == "":
            return "K"
        if element.VM > 1:
            new_values = [mappings.new_uid(v) for v in element.value]
            element.value = new_values
        else:
            element.value = mappings.new_uid(element.value)
        return "U"

    if action == "C":
        # in the baselines, C is only used by the date options: shift dates
        if element.VR in ["DA", "DT"]:
            if element.VM > 1:
                new_values = [shift_date_text(v, offset_days) for v in element.value]
                element.value = new_values
            else:
                element.value = shift_date_text(element.value, offset_days)
            return "C-shift"
        if element.VR == "TM":
            return "K"      # times are kept (no date in them)
        # other C (e.g. device descriptions) are kept in the baseline
        return "K"

    return "K"


def deid_dataset(ds, rules, mappings, patient_id, offset_days, counts):
    """Apply the rules to every element, also inside sequences."""
    all_tags = list(ds.keys())
    for tag in all_tags:
        element = ds[tag]
        tag_int = int(tag)

        if element.tag.is_private:
            action = "X"
        elif tag_int in rules:
            action = rules[tag_int][0]
        else:
            group_action = ps315.repeating_group_action(tag_int)
            if group_action is not None:
                action = group_action
            else:
                action = "K"   # not listed in the table -> kept (as in the standard)

        done = apply_action(ds, element, action, mappings, patient_id, offset_days)
        counts[done] = counts.get(done, 0) + 1

        # go inside sequences that were kept
        if done == "K" and element.VR == "SQ":
            for item in element.value:
                deid_dataset(item, rules, mappings, patient_id, offset_days, counts)


def ids_only(ds, mappings, patient_id, counts):
    """B0: change only the UIDs and the patient ID."""
    for element in ds.iterall():
        if element.VR == "UI" and element.keyword not in ["SOPClassUID", "ReferencedSOPClassUID",
                                                          "TransferSyntaxUID"]:
            if element.value in mappings.uids:
                element.value = mappings.uids[element.value]
                counts["U"] = counts.get("U", 0) + 1
    ds.PatientID = mappings.new_patient_id(patient_id)


def mark_deidentified(ds, baseline_name):
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "FP-DeID baseline " + baseline_name
    codes = METHOD_CODES[baseline_name]
    sequence = []
    for value, meaning in codes:
        item = Dataset()
        item.CodeValue = value
        item.CodingSchemeDesignator = "DCM"
        item.CodeMeaning = meaning
        sequence.append(item)
    if len(sequence) > 0:
        ds.DeidentificationMethodCodeSequence = Sequence(sequence)


# One file and a whole folder

# each worker process keeps its own copy of these (set once by start_worker)
WORKER = {}


def start_worker(baseline_name, rules, mappings, output_root):
    WORKER["baseline_name"] = baseline_name
    WORKER["rules"] = rules
    WORKER["mappings"] = mappings
    WORKER["output_root"] = output_root


def deid_file(input_path):
    baseline_name = WORKER["baseline_name"]
    rules = WORKER["rules"]
    mappings = WORKER["mappings"]
    output_root = WORKER["output_root"]
    counts = {}
    try:
        ds = pydicom.dcmread(input_path)
        patient_id = str(ds.get("PatientID", ""))
        offset_days = mappings.date_offset_days(patient_id)

        if baseline_name == "B0":
            ids_only(ds, mappings, patient_id, counts)
        else:
            deid_dataset(ds, rules, mappings, patient_id, offset_days, counts)
            mark_deidentified(ds, baseline_name)

        # the file meta header must agree with the new SOP Instance UID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        new_patient = str(ds.get("PatientID", "unknown"))
        new_series = str(ds.get("SeriesInstanceUID", "unknown"))
        folder = os.path.join(output_root, new_patient, new_series)
        os.makedirs(folder, exist_ok=True)
        output_path = os.path.join(folder, str(ds.SOPInstanceUID) + ".dcm")
        ds.save_as(output_path, enforce_file_format=False)
        return input_path, output_path, counts, ""
    except Exception as error:
        return input_path, "", counts, str(error)[:300]


def run_baseline(input_paths, output_root, baseline_name, mappings, workers=16):
    options = BASELINE_OPTIONS[baseline_name]
    rules = None
    if options is not None:
        rules = ps315.load_rules(options)
        # the scorer (and any research use) needs a pseudonymised Patient ID, never an empty one
        patient_id_tag = 0x00100020
        rules[patient_id_tag] = ("D", "pseudonym_override", "Patient ID")

    init_args = (baseline_name, rules, mappings, output_root)
    with Pool(workers, initializer=start_worker, initargs=init_args) as pool:
        results = pool.map(deid_file, input_paths, chunksize=20)

    rows = []
    for input_path, output_path, counts, error in results:
        row = {"input": input_path, "output": output_path, "error": error}
        for key in counts:
            row["n_" + key] = counts[key]
        rows.append(row)
    return pd.DataFrame(rows)
