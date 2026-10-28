"""DICOM PS3.15 Table E.1-1: which action each attribute gets."""

import os

import pandas as pd
from pydicom.datadict import dictionary_VR

from fpdeid import paths

TABLE_PATH = os.path.join(paths.CONFIGS, "ps315_table_e1_1.csv")

# columns of the table that are profile options
OPTION_COLUMNS = [
    "retain_safe_private",
    "retain_uids",
    "retain_device_id",
    "retain_institution_id",
    "retain_patient_chars",
    "retain_long_full_dates",
    "retain_long_modified_dates",
    "clean_descriptors",
    "clean_structured_content",
    "clean_graphics",
]


def pick_one(code):
    """Turn a combined code into one action.

    We keep the attribute (empty or dummy) instead of removing it when we can,
    because removing a required attribute makes the file invalid DICOM.
    """
    if code == "X/Z":
        return "Z"
    if code == "X/D":
        return "D"
    if code == "Z/D":
        return "D"
    if code == "X/Z/D":
        return "D"
    if code == "X/Z/U*":
        return "U"
    return code


def tag_to_int(tag_text):
    """'(0008,0050)' -> 0x00080050"""
    inside = tag_text.strip("()")
    group_text, element_text = inside.split(",")
    group = int(group_text, 16)
    element = int(element_text, 16)
    return (group << 16) + element


def load_rules(options):
    """Return {tag_int: (action, rule_name)} for the Basic Profile plus the chosen options.

    options: list of option column names, e.g. ["retain_long_modified_dates"]
    If several chosen options give an action for the same tag, the later one in the list wins.
    """
    table = pd.read_csv(TABLE_PATH, dtype=str).fillna("")
    rules = {}
    for _, row in table.iterrows():
        tag_text = row["tag"]
        if "x" in tag_text or "gggg" in tag_text:
            continue   # repeating groups and private tags are handled separately

        tag_int = tag_to_int(tag_text)
        action = pick_one(row["basic_profile"])

        # An empty sequence (Z) is invalid DICOM when the sequence must have items,
        # so for sequences "X/Z" means remove (X).
        if row["basic_profile"] == "X/Z" and is_sequence(tag_int):
            action = "X"
        rule_name = "basic_profile"

        for option in options:
            option_code = row[option]
            if option_code != "":
                action = option_code
                rule_name = option

        rules[tag_int] = (action, rule_name, row["name"])
    return rules


def is_sequence(tag_int):
    try:
        return dictionary_VR(tag_int) == "SQ"
    except KeyError:
        return False


def repeating_group_action(tag_int):
    """Curves (50xx,xxxx) and overlay data/comments (60xx,3000/4000) are removed."""
    group = tag_int >> 16
    element = tag_int & 0xFFFF
    if (group & 0xFF00) == 0x5000:
        return "X"
    if (group & 0xFF00) == 0x6000:
        if element == 0x3000 or element == 0x4000:
            return "X"
    return None
