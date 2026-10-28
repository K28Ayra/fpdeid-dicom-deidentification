"""Phase 2: policy-driven rules engine with an audit log.

For every element of a DICOM file the engine decides ONE action, applies it, and writes
an audit entry that says what was done, why, which PS3.15 option it implements and which
GDPR idea it serves.

Action codes (one letter each, used in the audit log):
    X = remove                  Z = empty value             D = dummy value
    P = pseudonym (from key)    S = date shift              T = time stamp shift (Unix seconds)
    K = keep (no audit entry)
    R = keep the sequence and apply the rules inside it     A = attribute added
"""

import csv
import datetime
import hashlib
import json
import os
import re
from multiprocessing import Pool

import pandas as pd
import pydicom
import yaml
from pydicom.datadict import dictionary_VR, tag_for_keyword
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from fpdeid import paths, ps315
from fpdeid.baselines import Mappings, shift_date_text

NUMBER_VRS = ["IS", "DS", "US", "SS", "UL", "SL", "FL", "FD", "SV", "UV"]
TEXT_VRS = ["LO", "SH", "ST", "LT", "UT", "UC"]

DEID_CODES = {
    "basic_profile": ("113100", "Basic Application Confidentiality Profile"),
    "retain_long_modified_dates": ("113107", "Retain Longitudinal Temporal Information Modified Dates Option"),
    "retain_patient_chars": ("113108", "Retain Patient Characteristics Option"),
    "retain_safe_private": ("113111", "Retain Safe Private Option"),
    "clean_descriptors": ("113105", "Clean Descriptors Option"),
}


# ---------------------------------------------------------------
# Loading the policy
# ---------------------------------------------------------------

def load_safe_private():
    """Read PS3.15 Table E.3.10-1 into a set of (group, creator, last two hex digits)."""
    safe = set()
    table_path = os.path.join(paths.CONFIGS, "ps315_safe_private.csv")
    with open(table_path) as f:
        for row in csv.DictReader(f):
            tag_text = row["tag"].strip("()")
            group_text, element_text = tag_text.split(",")
            group = int(group_text, 16)
            last_two = int(element_text[-2:], 16)
            creator = row["creator"].strip().upper()
            safe.add((group, creator, last_two))
    return safe


class Policy:
    """Everything the engine needs, read once from configs/policy.yaml."""

    def __init__(self, policy_path, uid_map_csv, patient_map_csv):
        with open(policy_path) as f:
            self.config = yaml.safe_load(f)

        self.name = self.config["name"]
        self.options = self.config.get("options", [])
        self.rules = ps315.load_rules(self.options)
        self.mappings = Mappings(uid_map_csv, patient_map_csv)
        self.remap_known_uids = self.config["pseudonym_key"].get("remap_every_known_uid", False)

        private = self.config["private_tags"]
        self.private_keep_vrs = set(private["keep_vrs"])
        self.private_shift_vrs = set(private["shift_vrs"])
        self.private_remap_vrs = set(private["remap_vrs"])
        self.safe_private = set()
        if private.get("keep_safe_private_table", False):
            self.safe_private = load_safe_private()

        self.date_check = self.config["value_checks"].get("numbers_that_look_like_dates", "")

        self.keep_and_clean = set()
        for keyword in self.config.get("keep_and_clean_sequences", []):
            self.keep_and_clean.add(tag_for_keyword(keyword))

        self.overrides = {}
        for item in self.config.get("overrides", []):
            tag_int = tag_for_keyword(item["keyword"])
            self.overrides[tag_int] = (item["action"], item["reason"])

        self.add_if_missing = self.config.get("add_if_missing", {})
        self.fill_if_missing = self.config.get("fill_if_missing", {})
        self.gdpr = self.config["gdpr"]

        # Phase 3: the PHI fingerprint (the fingerprints themselves are added with set_fingerprints)
        settings = self.config.get("fingerprint", {})
        self.use_fingerprint = settings.get("enabled", False)
        self.clean_all_kept_text = settings.get("clean_all_kept_text", False)
        self.common_word_share = settings.get("common_word_share", 0.02)
        self.private_clean_vrs = set(private.get("clean_vrs", []))
        self.fingerprints = {}
        self.shared_cleaner = None
        self.empty_fingerprint = None
        if self.use_fingerprint:
            from fpdeid.fingerprint import Fingerprint
            self.empty_fingerprint = Fingerprint([])     # generic patterns only, for unknown patients
        self.private_id_words = private.get("remove_if_name_has", [])
        self.site_level = settings.get("site_level", False)
        # coded values (controlled vocabulary) are never changed by the text cleaner
        self.never_clean = set(["CodeValue", "CodingSchemeDesignator", "CodingSchemeVersion", "CodeMeaning",
                                "LongCodeValue", "URNCodeValue", "MappingResource", "ContextIdentifier"])
        if settings.get("never_clean_research_fields", False):
            with open(os.path.join(paths.CONFIGS, "utility_tags.yaml")) as f:
                for group in yaml.safe_load(f).values():
                    self.never_clean.update(group)
        self.clean_readable_unknown = private.get("readable_unknown", "") == "clean"

    def set_fingerprints(self, fingerprints):
        """fingerprints: {original PatientID: Fingerprint} from fingerprint.build_all()."""
        from fpdeid.fingerprint import Fingerprint
        self.fingerprints = fingerprints
        self.empty_fingerprint = Fingerprint([])     # generic patterns only, for unknown patients

    def set_shared_cleaner(self, cleaner):
        """Use one cleaner for every patient (used for the NER comparison in the ablation study)."""
        self.shared_cleaner = cleaner

    def fingerprint_for(self, patient_id):
        if self.shared_cleaner is not None:
            return self.shared_cleaner
        if patient_id in self.fingerprints:
            return self.fingerprints[patient_id]
        return self.empty_fingerprint


# ---------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------

def looks_like_date(value):
    """True for 8-digit numbers like 20160807 (years 1900-2099, valid month and day)."""
    text = str(value).strip()
    if len(text) != 8 or not text.isdigit():
        return False
    if text[:2] not in ["19", "20"]:
        return False
    try:
        datetime.datetime.strptime(text, "%Y%m%d")
    except ValueError:
        return False
    return True


# Unix time (seconds since 1970) between 1990-01-01 and 2030-01-01
EPOCH_FIRST = 631152000
EPOCH_LAST = 1893456000
INTEGER_VRS = ["IS", "SL", "UL", "SV", "UV"]


def looks_like_timestamp(element, value):
    """True for an integer that is a plausible Unix time, in a field whose name talks about time."""
    if element.VR not in INTEGER_VRS:
        return False
    name = str(element.name).lower()
    if "time" not in name and "date" not in name and "stamp" not in name:
        return False
    try:
        number = int(str(value))
    except ValueError:
        return False
    return EPOCH_FIRST <= number <= EPOCH_LAST


def shift_timestamp(element, offset_days):
    seconds = offset_days * 24 * 60 * 60
    if element.VM > 1:
        return [int(v) + seconds for v in element.value]
    return int(element.value) + seconds


def short_hash(value):
    """The audit log stores only a hash of the old value, so the log itself does not leak PHI."""
    text = str(value)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def tag_text(element):
    return "(%04X,%04X)" % (element.tag.group, element.tag.element)


def name_says_identifier(element, phrases):
    """True if the (vendor) name of a private element contains a phrase like 'serial' or 'suite id'."""
    name = " ".join(re.findall(r"[a-z]+", str(element.name).lower()))
    for phrase in phrases:
        if re.search(r"\b" + re.escape(phrase) + r"\b", name):
            return True
    return False


def readable_text(element):
    """For UN/OB values: the text if the bytes are plain readable characters, else None."""
    try:
        raw = bytes(element.value)
    except Exception:
        return None
    if len(raw) == 0:
        return None
    text = raw.decode("latin-1").rstrip("\x00 ")
    if text == "":
        return None
    for character in text:
        if not (character.isprintable() or character in "\\"):
            return None
    return text


def is_private_creator(element):
    return element.tag.is_private and 0x0010 <= element.tag.element <= 0x00FF


def private_key(element):
    """(group, creator, last two hex digits) for a private element, used for the safe list."""
    creator = str(element.private_creator or "").strip().upper()
    last_two = element.tag.element & 0xFF
    return (element.tag.group, creator, last_two)


# ---------------------------------------------------------------
# Decide the action for one element
# ---------------------------------------------------------------

def decide(element, policy):
    """Return (action, reason, ps315_option)."""
    tag_int = int(element.tag)
    vr = element.VR

    # --- private tags: decided by value type ---
    if element.tag.is_private:
        if is_private_creator(element):
            return "K", "private creator (checked again after its block)", "retain_safe_private"
        if private_key(element) in policy.safe_private:
            return "K", "listed in PS3.15 Table E.3.10-1 (safe private)", "retain_safe_private"
        if vr == "SQ":
            return "R", "private sequence: rules applied inside", "retain_safe_private"
        if vr in policy.private_keep_vrs:
            return "K", "private number/code value (VR " + vr + ")", "retain_safe_private"
        if vr in policy.private_shift_vrs:
            return "S", "private date", "retain_long_modified_dates"
        if vr in policy.private_remap_vrs:
            return "P", "private UID", "basic_profile"
        readable = vr in ["UN", "OB"] and policy.clean_readable_unknown and readable_text(element) is not None
        if (vr in policy.private_clean_vrs or readable) and policy.use_fingerprint:
            if name_says_identifier(element, policy.private_id_words):
                return "X", "private text whose name says it is an identifier: " + str(element.name), "basic_profile"
            return "F", "private text checked with the PHI fingerprint", "retain_safe_private"
        return "X", "private name/text/binary value (VR " + vr + "), may hold PHI", "basic_profile"

    # --- explicit overrides in the policy ---
    if tag_int in policy.overrides:
        action, reason = policy.overrides[tag_int]
        if action == "D" and element.keyword == "PatientID":
            return "P", reason, "basic_profile"
        return action, reason, "policy_override"

    # --- sequences that we keep and clean inside ---
    if vr == "SQ" and tag_int in policy.keep_and_clean:
        return "R", "sequence kept, rules applied inside", "clean_structured_content"

    # --- every UID that is in the pseudonym key ---
    if vr == "UI" and policy.remap_known_uids:
        value = str(element.value).strip()
        if value in policy.mappings.uids:
            return "P", "UID found in pseudonym key", "basic_profile"

    # --- the PS3.15 table (with the chosen options) ---
    if tag_int in policy.rules:
        code, rule_name, name = policy.rules[tag_int]
        clean_text = policy.use_fingerprint and rule_name == "clean_descriptors"
        action = action_from_code(code, element, clean_text)
        return action, "PS3.15 Table E.1-1: " + name, rule_name

    group_action = ps315.repeating_group_action(tag_int)
    if group_action is not None:
        return group_action, "PS3.15: curve/overlay data", "basic_profile"

    return "K", "not listed in PS3.15 Table E.1-1", "basic_profile"


def action_from_code(code, element, use_fingerprint=False):
    """Turn a PS3.15 code (X, Z, D, U, C, K) into our action letter."""
    vr = element.VR
    if vr == "SQ":
        if code in ["U", "C", "D", "K"]:
            return "R"
        if code == "Z":
            return "Z"
        return "X"
    if code == "U":
        return "P"
    if code == "C":
        if vr in ["DA", "DT"]:
            return "S"
        if vr == "TM":
            return "K"
        if use_fingerprint and vr in TEXT_VRS:
            return "F"  # clean the text: remove only what matches the PHI fingerprint
        return "X"      # without the fingerprint, text that must be "cleaned" is removed
    if code == "D" and element.keyword in ["PatientID", "PatientName"]:
        return "P"
    return code


# ---------------------------------------------------------------
# Apply one action
# ---------------------------------------------------------------

def dummy_for(element):
    vr = element.VR
    if vr == "DA":
        return "19000101"
    if vr == "TM":
        return "000000"
    if vr == "DT":
        return "19000101000000"
    if vr in NUMBER_VRS:
        return 0
    if vr == "AS":
        return "000Y"
    return "ANONYMIZED"


def pseudonym_for(element, policy, patient_id):
    if element.keyword in ["PatientID", "PatientName"]:
        return policy.mappings.new_patient_id(patient_id)
    if element.VM > 1:
        return [policy.mappings.new_uid(v) for v in element.value]
    return policy.mappings.new_uid(element.value)


def shift_value(element, offset_days):
    """Shift a date (DA/DT) or a number that is really a date."""
    if element.VM > 1:
        values = list(element.value)
    else:
        values = [element.value]

    new_values = []
    for value in values:
        shifted = shift_date_text(value, offset_days)
        if element.VR in NUMBER_VRS:
            shifted = int(shifted) if shifted != "" else 0
        new_values.append(shifted)

    if element.VM > 1:
        return new_values
    return new_values[0]


def apply(ds, element, action, policy, patient_id, offset_days):
    """Change the element in place. Returns the new value as text (for the audit log)."""
    if action == "X":
        del ds[element.tag]
        return ""
    if action == "Z":
        if element.VR == "SQ":
            element.value = Sequence([])
        else:
            element.value = ""
        return ""
    if action == "D":
        element.value = dummy_for(element)
        return str(element.value)
    if action == "P":
        if element.value is None or str(element.value) == "":
            return ""
        element.value = pseudonym_for(element, policy, patient_id)
        return str(element.value)
    if action == "S":
        if element.value is None or str(element.value) == "":
            return ""
        element.value = shift_value(element, offset_days)
        return str(element.value)
    if action == "T":
        element.value = shift_timestamp(element, offset_days)
        return str(element.value)
    return str(element.value)


def clean_text_element(element, policy, patient_id):
    """Remove the fingerprint matches from a text element.

    Returns None if nothing matched, else (old_hash, new_value_text, list of "category from source").
    """
    if element.value is None or str(element.value) == "":
        return None
    fingerprint = policy.fingerprint_for(patient_id)

    is_bytes = element.VR in ["UN", "OB"]
    if is_bytes:
        values = [readable_text(element)]
    elif element.VM > 1:
        values = [str(v) for v in element.value]
    else:
        values = [str(element.value)]

    new_values = []
    matches = []
    for value in values:
        cleaned, found = fingerprint.clean(value)
        new_values.append(cleaned)
        for category, source in found:
            label = category + " (" + source + ")"
            if label not in matches:
                matches.append(label)

    if len(matches) == 0:
        return None

    old_hash = short_hash(element.value)
    if is_bytes:
        new_bytes = new_values[0].encode("latin-1")
        if len(new_bytes) % 2 == 1:
            new_bytes = new_bytes + b" "          # DICOM values have an even length
        element.value = new_bytes
        return old_hash, new_values[0], matches
    if element.VM > 1:
        element.value = new_values
    else:
        element.value = new_values[0]
    return old_hash, str(element.value), matches


# ---------------------------------------------------------------
# Walk a dataset (recursively) and write the audit log
# ---------------------------------------------------------------

def clean_dataset(ds, policy, patient_id, offset_days, audit, path):
    used_blocks_before = private_blocks_in_use(ds)
    all_tags = list(ds.keys())
    for tag in all_tags:
        element = ds[tag]
        action, reason, option = decide(element, policy)

        # value check: a kept number that is really a date is shifted
        if action == "K" and element.VR in NUMBER_VRS and policy.date_check == "shift":
            values = element.value if element.VM > 1 else [element.value]
            if any(looks_like_date(v) for v in values):
                action = "S"
                reason = "number that looks like a date (value check)"
                option = "retain_long_modified_dates"
            elif any(looks_like_timestamp(element, v) for v in values):
                action = "T"
                reason = "number that looks like a Unix time stamp (value check)"
                option = "retain_long_modified_dates"

        # a kept sequence is never trusted blindly: the rules are always applied inside it
        if action == "K" and element.VR == "SQ":
            action = "R"

        # safety net: every kept text value also goes through the PHI fingerprint
        if action == "K" and element.VR in TEXT_VRS and policy.use_fingerprint and policy.clean_all_kept_text:
            if not is_private_creator(element) and element.keyword not in policy.never_clean:
                action = "F"
                reason = "kept text checked with the PHI fingerprint"
                option = "clean_descriptors"

        if action == "F":
            changed = clean_text_element(element, policy, patient_id)
            if changed is None:
                continue        # nothing identifying found: the value stays exactly as it was
            old_hash, new_value, matches = changed
            audit.append({
                "tag": path + tag_text(element),
                "keyword": element.keyword or element.name,
                "VR": element.VR,
                "action": "F",
                "reason": reason + "; removed: " + ", ".join(matches),
                "ps315_option": option,
                "gdpr": policy.gdpr["F"],
                "old_value_hash": old_hash,
                "new_value": new_value[:64],
            })
            continue

        if action == "K":
            continue

        if action == "R":
            item_path = path + tag_text(element)
            for item in element.value:
                clean_dataset(item, policy, patient_id, offset_days, audit, item_path + ">")
            continue

        old_hash = short_hash(element.value)
        new_value = apply(ds, element, action, policy, patient_id, offset_days)
        gdpr = policy.gdpr.get(action, "")
        entry = {
            "tag": path + tag_text(element),
            "keyword": element.keyword or element.name,
            "VR": element.VR,
            "action": action,
            "reason": reason,
            "ps315_option": option,
            "gdpr": gdpr,
            "old_value_hash": old_hash,
            "new_value": new_value[:64],
        }
        audit.append(entry)

    remove_empty_private_blocks(ds, policy, audit, path, used_blocks_before)


def private_blocks_in_use(ds):
    """Set of (group, block) of private blocks that contain at least one element."""
    used = set()
    for element in ds:
        if element.tag.is_private and not is_private_creator(element):
            used.add((element.tag.group, element.tag.element >> 8))
    return used


def remove_empty_private_blocks(ds, policy, audit, path, used_blocks_before):
    """A private creator is removed only if WE emptied its block (a block that was empty already stays)."""
    used_now = private_blocks_in_use(ds)
    creators = [e for e in ds if is_private_creator(e)]
    for creator in creators:
        block_key = (creator.tag.group, creator.tag.element)
        emptied_by_us = block_key in used_blocks_before and block_key not in used_now
        if emptied_by_us:
            audit.append({
                "tag": path + tag_text(creator),
                "keyword": "PrivateCreator",
                "VR": creator.VR,
                "action": "X",
                "reason": "private creator of an empty block",
                "ps315_option": "basic_profile",
                "gdpr": policy.gdpr["X"],
                "old_value_hash": short_hash(creator.value),
                "new_value": "",
            })
            del ds[creator.tag]


def add_attribute(ds, keyword, value, reason, policy, audit):
    tag_int = tag_for_keyword(keyword)
    vr = dictionary_VR(tag_int)
    ds.add_new(tag_int, vr, value)
    audit.append({
        "tag": "(%04X,%04X)" % (tag_int >> 16, tag_int & 0xFFFF),
        "keyword": keyword,
        "VR": vr,
        "action": "A",
        "reason": reason,
        "ps315_option": "basic_profile",
        "gdpr": policy.gdpr["A"],
        "old_value_hash": "",
        "new_value": "" if value is None else str(value),
    })


def add_missing(ds, policy, audit):
    # Type 2 attributes: must be present, may be empty
    modality = str(ds.get("Modality", ""))
    wanted = policy.add_if_missing.get(modality, [])
    for keyword in wanted:
        if keyword not in ds:
            reason = "Type 2 attribute required by the " + modality + " IOD, added empty"
            add_attribute(ds, keyword, None, reason, policy, audit)

    # attributes whose value follows from the SOP Class
    sop_class = str(ds.get("SOPClassUID", ""))
    fill = policy.fill_if_missing.get(sop_class, {})
    for keyword in fill:
        # fill when missing, and also when present but empty
        if keyword in ds and str(ds.data_element(keyword).value or "") != "":
            continue
        if keyword in ds:
            del ds[keyword]
        value = fill[keyword]
        if value == "from_photometric":
            if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
                value = "INVERSE"
            else:
                value = "IDENTITY"
        if value == "":
            value = None
        reason = "required by the IOD of SOP Class " + sop_class + ", value follows from the SOP Class"
        add_attribute(ds, keyword, value, reason, policy, audit)


def mark_deidentified(ds, policy, audit):
    for keyword in ["PatientIdentityRemoved", "DeidentificationMethod", "DeidentificationMethodCodeSequence"]:
        tag_int = tag_for_keyword(keyword)
        audit.append({
            "tag": "(%04X,%04X)" % (tag_int >> 16, tag_int & 0xFFFF),
            "keyword": keyword,
            "VR": dictionary_VR(tag_int),
            "action": "A",
            "reason": "records that and how the file was de-identified (PS3.15 E.1.1)",
            "ps315_option": "basic_profile",
            "gdpr": policy.gdpr["A"],
            "old_value_hash": short_hash(ds[tag_int].value) if tag_int in ds else "",
            "new_value": "",
        })
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = policy.name
    names = ["basic_profile"] + policy.options + ["retain_safe_private"]
    items = []
    for name in names:
        if name not in DEID_CODES:
            continue
        value, meaning = DEID_CODES[name]
        item = Dataset()
        item.CodeValue = value
        item.CodingSchemeDesignator = "DCM"
        item.CodeMeaning = meaning
        items.append(item)
    ds.DeidentificationMethodCodeSequence = Sequence(items)


# ---------------------------------------------------------------
# One file, and a whole list of files
# ---------------------------------------------------------------

WORKER = {}


def start_worker(policy, output_root):
    WORKER["policy"] = policy
    WORKER["output_root"] = output_root


def clean_file(input_path):
    policy = WORKER["policy"]
    output_root = WORKER["output_root"]
    audit = []
    try:
        ds = pydicom.dcmread(input_path)
        patient_id = str(ds.get("PatientID", ""))
        offset_days = policy.mappings.date_offset_days(patient_id)

        clean_dataset(ds, policy, patient_id, offset_days, audit, "")
        add_missing(ds, policy, audit)
        mark_deidentified(ds, policy, audit)
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        folder = os.path.join(output_root, str(ds.PatientID), str(ds.SeriesInstanceUID))
        os.makedirs(folder, exist_ok=True)
        output_path = os.path.join(folder, str(ds.SOPInstanceUID) + ".dcm")
        ds.save_as(output_path, enforce_file_format=False)
        error = ""
    except Exception as problem:
        output_path = ""
        error = str(problem)[:300]

    record = {"input": input_path, "output": output_path, "error": error, "changes": audit}
    return record


def run(input_paths, output_root, policy, workers=16):
    """Clean all files. Writes <output_root>/audit.jsonl (one line per file) and returns a summary table."""
    os.makedirs(output_root, exist_ok=True)
    audit_path = os.path.join(output_root, "audit.jsonl")
    rows = []
    with open(audit_path, "w") as audit_file:
        with Pool(workers, initializer=start_worker, initargs=(policy, output_root)) as pool:
            for record in pool.imap(clean_file, input_paths, chunksize=20):
                audit_file.write(json.dumps(record) + "\n")
                counts = {}
                for change in record["changes"]:
                    counts[change["action"]] = counts.get(change["action"], 0) + 1
                row = {"input": record["input"], "output": record["output"], "error": record["error"]}
                for action in counts:
                    row["n_" + action] = counts[action]
                rows.append(row)
    return pd.DataFrame(rows)
