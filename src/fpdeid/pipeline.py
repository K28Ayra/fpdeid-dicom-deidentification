"""The whole FP-DeID pipeline for one data split (Phase 5): fingerprints -> headers -> pixels.

The rules are frozen: configs/policy.yaml, the same code as Phase 4. Only the pseudonym key changes
(each split has its own patient and UID mapping tables).
"""

import json
import os
import shutil
import time

import pandas as pd

from fpdeid import engine
from fpdeid import fingerprint
from fpdeid import paths
from fpdeid import pixels

SPLITS = {
    "validation": {
        "file_table": os.path.join(paths.DERIVED, "validation_file_table.csv"),
        "uid_map": paths.VALIDATION_UID_MAP,
        "patient_map": paths.VALIDATION_PATIENT_MAP,
        "answer_db": os.path.join(paths.VALIDATION_ANSWER_DIR, "MIDI-B-Answer-Key-Validation.db"),
    },
    "test": {
        "file_table": os.path.join(paths.DERIVED, "test_file_table.csv"),
        "uid_map": paths.TEST_UID_MAP,
        "patient_map": paths.TEST_PATIENT_MAP,
        "answer_db": os.path.join(paths.TEST_ANSWER_DIR, "MIDI-B-Answer-Key-Test.db"),
    },
}


def load_file_table(split):
    table = pd.read_csv(SPLITS[split]["file_table"], dtype=str)
    table = table[table["readable"] == "True"]
    return table


def run_split(split, out_folder, workers=16):
    """De-identify every file of a split into out_folder. Writes audit.jsonl, audit_pixels.jsonl, summary.json."""
    settings = SPLITS[split]
    file_table = load_file_table(split)
    all_files = file_table["path"].tolist()
    patient_of = dict(zip(file_table["path"], file_table["patient_id"]))
    modality_of = dict(zip(file_table["path"], file_table["modality"]))
    series_of = dict(zip(file_table["path"], file_table["series_uid"]))

    policy_path = os.path.join(paths.CONFIGS, "policy.yaml")
    policy = engine.Policy(policy_path, settings["uid_map"], settings["patient_map"])

    summary = {"split": split, "files": len(all_files), "patients": file_table["patient_id"].nunique()}

    start = time.time()
    fingerprints, common_words = fingerprint.build_all(all_files, workers=workers,
                                                       common_share=policy.common_word_share,
                                                       use_site=policy.site_level)
    policy.set_fingerprints(fingerprints)
    summary["fingerprint seconds"] = round(time.time() - start, 1)

    shutil.rmtree(out_folder, ignore_errors=True)
    start = time.time()
    run_table = engine.run(all_files, out_folder, policy, workers=workers)
    summary["header seconds"] = round(time.time() - start, 1)
    summary["header errors"] = int((run_table["error"] != "").sum())

    records = []
    with open(os.path.join(out_folder, "audit.jsonl")) as f:
        for line in f:
            record = json.loads(line)
            if record["output"] != "":
                records.append({"input": record["input"], "output": record["output"]})

    pixel_audit = os.path.join(out_folder, "audit_pixels.jsonl")
    pixel_summary = pixels.clean_outputs(records, fingerprints, patient_of, modality_of, series_of, pixel_audit,
                                         method="fpdeid")
    summary.update(pixel_summary)

    with open(os.path.join(out_folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def auditor_jobs(split, out_folder):
    """[(original files of a patient, output files of that patient)] from the run's audit log."""
    file_table = load_file_table(split)
    patient_of = dict(zip(file_table["path"], file_table["patient_id"]))
    originals = {}
    outputs = {}
    with open(os.path.join(out_folder, "audit.jsonl")) as f:
        for line in f:
            record = json.loads(line)
            if record["output"] == "":
                continue
            patient = patient_of[record["input"]]
            originals.setdefault(patient, []).append(record["input"])
            outputs.setdefault(patient, []).append(record["output"])
    jobs = []
    for patient in sorted(originals):
        jobs.append((originals[patient], outputs[patient]))
    return jobs


def audit_and_fix(split, out_folder, workers=16):
    """Closed loop: the independent auditor scans every output header; every hit is removed and logged."""
    from fpdeid import auditor
    jobs = auditor_jobs(split, out_folder)
    hits, elements = auditor.audit_headers(jobs, workers=workers)
    log_path = os.path.join(out_folder, "audit_auditor_fixes.jsonl")
    open(log_path, "w").close()
    fixed_files = auditor.fix_header_hits(hits, log_path)
    hits_after, elements_after = auditor.audit_headers(jobs, workers=workers)
    result = {"auditor elements read": elements, "auditor hits": len(hits), "auditor fixed files": fixed_files,
              "auditor hits after fix": len(hits_after)}
    with open(os.path.join(out_folder, "auditor_hits.json"), "w") as f:
        json.dump(hits, f)
    return result
