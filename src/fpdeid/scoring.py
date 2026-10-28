"""Run the official MIDI-B validation script and read its results."""

import json
import os
import sqlite3
import subprocess

import pandas as pd

from fpdeid import paths

WEIGHTS = {"hipaa": 0.70, "dicom": 0.20, "tcia": 0.10}


def run_official_scorer(run_name, deid_folder, answer_db, uid_map, patient_map, out_folder, cpus=16):
    """Write a config file and run run_validation.py. Returns the path of validation_results.db."""
    os.makedirs(out_folder, exist_ok=True)
    config = {
        "run_name": run_name,
        "input_data_path": deid_folder,
        "output_data_path": out_folder,
        "answer_db_file": answer_db,
        "uid_mapping_file": uid_map,
        "patid_mapping_file": patient_map,
        "multiprocessing": "True",
        "multiprocessing_cpus": str(cpus),
        "log_path": os.path.join(out_folder, "logs"),
        "log_level": "warning",
        "report_series": "False",
    }
    config_path = os.path.join(out_folder, run_name + "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # keep the scorer's downloads (NLTK words, EasyOCR model) on /home, and use CPU only
    env = dict(os.environ)
    env["NLTK_DATA"] = os.path.join(paths.PROJECT, "tools", "nltk_data")
    env["EASYOCR_MODULE_PATH"] = os.path.join(paths.PROJECT, "tools", "easyocr")
    env["CUDA_VISIBLE_DEVICES"] = ""

    command = [paths.SCORER_PYTHON, "run_validation.py", config_path]
    log_file = os.path.join(out_folder, run_name + "_scorer_stdout.log")
    with open(log_file, "w") as log:
        subprocess.run(command, cwd=paths.SCORER_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)

    db_path = os.path.join(out_folder, run_name, "validation_results.db")
    return db_path


def load_results(db_path):
    connection = sqlite3.connect(db_path)
    table = pd.read_sql("SELECT * FROM validation_results", connection)
    connection.close()
    return table


def check_category(row):
    """Copy of reports_helper.check_category in the official script.

    Returns 'hipaa', 'dicom', 'tcia' or 'unknown'.
    """
    action = row["action"]
    if action in ["<tag_retained>", "<text_notnull>"]:
        return "dicom"
    if action in ["<date_shifted>", "<uid_changed>", "<pixels_hidden>"]:
        return "hipaa"
    if action in ["<patid_consistent>", "<uid_consistent>"]:
        return "dicom"
    if action in ["<pixels_retained>"]:
        return "tcia"
    if action == "<text_removed>":
        if row["hipaa_m"] or row["hipaa_z"]:
            return "hipaa"
        if row["tcia_p15"] or row["tcia_ptkb"] or row["tcia_rev"]:
            return "tcia"
    if action == "<text_retained>":
        if row["tcia_p15"] or row["tcia_ptkb"] or row["tcia_rev"]:
            return "tcia"
    return "unknown"


def add_categories(table):
    table = table.copy()
    categories = []
    for _, row in table.iterrows():
        categories.append(check_category(row))
    table["score_cat"] = categories
    return table


def summarise(table):
    """Official-style weighted score + simple pass rates.

    check_passed: 1 = pass, 0 = fail, missing = the scorer could not evaluate it (counted as not passed).
    """
    if "score_cat" not in table.columns:
        table = add_categories(table)
    passed = table["check_passed"].fillna(0).astype(float)
    table = table.copy()
    table["passed"] = passed

    summary = {}
    summary["checks"] = len(table)
    summary["plain_accuracy_%"] = round(100 * passed.mean(), 3)

    weighted = 0.0
    for name in ["hipaa", "dicom", "tcia"]:
        part = table[table["score_cat"] == name]
        if len(part) == 0:
            continue
        accuracy = part["passed"].mean()
        summary[name + "_checks"] = len(part)
        summary[name + "_accuracy_%"] = round(100 * accuracy, 3)
        weighted = weighted + WEIGHTS[name] * accuracy
    summary["weighted_score_%"] = round(100 * weighted, 3)
    return summary


# Our own view of the checks: privacy vs utility vs consistency

# a failed privacy check = an identifier left in the output (a leak)
PRIVACY_ACTIONS = ["<text_removed>", "<date_shifted>", "<uid_changed>", "<pixels_hidden>"]
# a failed utility check = useful data removed or damaged (over-cleaning)
UTILITY_ACTIONS = ["<text_retained>", "<tag_retained>", "<text_notnull>", "<pixels_retained>"]
# a failed consistency check = pseudonym not the expected one
CONSISTENCY_ACTIONS = ["<uid_consistent>", "<patid_consistent>"]


def privacy_utility(table):
    passed = table["check_passed"].fillna(0).astype(float)
    result = {}
    groups = [("privacy", PRIVACY_ACTIONS), ("utility", UTILITY_ACTIONS), ("consistency", CONSISTENCY_ACTIONS)]
    for name, actions in groups:
        in_group = table["action"].isin(actions)
        n_checks = int(in_group.sum())
        n_passed = int(passed[in_group].sum())
        result[name + "_checks"] = n_checks
        result[name + "_failed"] = n_checks - n_passed
        if n_checks > 0:
            result[name + "_pass_%"] = round(100 * n_passed / n_checks, 3)
    return result


def per_action(table):
    table = table.copy()
    table["passed"] = table["check_passed"].fillna(0).astype(float)
    grouped = table.groupby("action")["passed"].agg(["count", "sum"])
    grouped["failed"] = grouped["count"] - grouped["sum"]
    grouped["accuracy_%"] = (100 * grouped["sum"] / grouped["count"]).round(3)
    grouped = grouped.rename(columns={"count": "checks", "sum": "passed"})
    return grouped


def top_failures(table, n=20):
    failed = table[table["check_passed"].fillna(0).astype(float) < 1]
    grouped = failed.groupby(["action", "tag_name"]).size()
    grouped = grouped.sort_values(ascending=False).head(n)
    return grouped.reset_index(name="failures")


def make_answer_subset(answer_db, sop_uids, subset_db):
    """Write a smaller answer key that only has the given (original) SOP Instance UIDs.

    Useful for fast runs on dev-mini: the scorer counts every answer row without an output
    file as a "missing file", so a run on 20 patients must use a 20-patient answer key.
    """
    if os.path.exists(subset_db):
        os.remove(subset_db)
    wanted = set(sop_uids)
    source = sqlite3.connect(answer_db)
    target = sqlite3.connect(subset_db)
    first = True
    for chunk in pd.read_sql("SELECT * FROM answer_data", source, chunksize=2000):
        keep = chunk[chunk["SOPInstanceUID"].isin(wanted)]
        if len(keep) == 0:
            continue
        if first:
            keep.to_sql("answer_data", target, index=False, if_exists="replace")
            first = False
        else:
            keep.to_sql("answer_data", target, index=False, if_exists="append")
    source.close()
    target.close()
    return subset_db


# Why did a check fail? (one cause per failed check, used in Phases 3 and 5)

ICON_TAGS = ["<Rows>", "<Columns>", "<Bits Stored>", "<Bits Allocated>", "<High Bit>", "<Samples per Pixel>",
             "<Pixel Representation>", "<Photometric Interpretation>", "<Smallest Image Pixel Value>",
             "<Largest Image Pixel Value>"]


def free_text_names():
    """Scorer names of the descriptor fields that PS3.15 'Clean Descriptors' cleans, plus SR text."""
    table = pd.read_csv(os.path.join(paths.CONFIGS, "ps315_table_e1_1.csv"), dtype=str).fillna("")
    names = ["<" + name + ">" for name in table[table["clean_descriptors"] == "C"]["name"]]
    names.append("<Text Value>")
    return names


def failure_cause(row, free_text):
    tag_name = str(row["tag_name"])
    tag_ds = str(row["tag_ds"])
    if row["action"] == "<pixels_hidden>":
        return "burned-in text in pixels not hidden"
    if row["action"] == "<pixels_retained>":
        return "pixels changed although no PHI there"
    if tag_ds.count('"') >= 4:
        return "private tag inside a private sequence: the scorer cannot read it (also fails for unchanged files)"
    is_private = tag_name.startswith("<[") or "Private" in tag_name
    if is_private and row["action"] == "<text_removed>":
        return "private value removed by TCIA's vendor knowledge base (not PHI)"
    if tag_name == "<Private Creator>":
        return "label of a private block whose content was binary and removed"
    if is_private and str(row["file_value"]) == "<>":
        return "private value empty in the input"
    if is_private and row["action"] == "<text_retained>" and str(row["file_value"]) == "None":
        return "private binary value removed (cannot be checked)"
    if "Palette" in tag_name or tag_name in ICON_TAGS:
        return "icon image removed on purpose (may show burned-in text)"
    if row["action"] == "<text_removed>":
        return "LEAK: identifier not found by fingerprint or patterns"
    if row["action"] in ["<date_shifted>", "<uid_changed>"]:
        return "LEAK: " + row["action"].strip("<>").replace("_", " ") + " check failed"
    if row["action"] == "<text_retained>" and tag_name in free_text:
        return "over-cleaning in free text"
    if is_private:
        return "over-cleaning in private text"
    if row["action"] in ["<uid_consistent>", "<patid_consistent>"]:
        return "pseudonym not the expected one"
    return "other: " + row["action"]


def failures_by_cause(table):
    passed = table["check_passed"].fillna(0).astype(float)
    failed = table[passed < 1].copy()
    free_text = free_text_names()
    causes = []
    for row in failed.to_dict("records"):
        causes.append(failure_cause(row, free_text))
    failed["cause"] = causes
    return failed
