"""Integration tests of the one-command pipeline (Phase 6) on 3 tiny synthetic files in tests/data/input.

They need no GPU and no MIDI-B data, so they also run on GitHub Actions. The pixel step is skipped (--no-pixels).
"""

import glob
import hashlib
import json
import os

import pydicom

from fpdeid import paths
from fpdeid import run

INPUT = os.path.join(os.path.dirname(__file__), "data", "input")
SECRETS = ["TESTER", "ALICE", "SAMPLE", "A1234567", "B7654321", "19700412", "19581130", "20150807", "20190302",
           "555-123-4567", "ACC0001", "HOUSE", "1.2.826.0.1.111"]


def run_once(folder):
    return run.run(INPUT, str(folder), os.path.join(paths.CONFIGS, "policy.yaml"), workers=2,
                   do_pixels=False)


def output_hashes(folder):
    hashes = {}
    for path in glob.glob(os.path.join(str(folder), "*", "*", "*.dcm")):
        relative = os.path.relpath(path, str(folder))
        with open(path, "rb") as f:
            hashes[relative] = hashlib.sha256(f.read()).hexdigest()
    return hashes


def test_no_identifier_left_anywhere_in_the_bytes(tmp_path):
    summary = run_once(tmp_path)
    assert summary["files"] == 3 and summary["header errors"] == 0
    for path in glob.glob(os.path.join(str(tmp_path), "*", "*", "*.dcm")):
        with open(path, "rb") as f:
            content = f.read().decode("latin-1").upper()
        for secret in SECRETS:
            assert secret.upper() not in content, secret + " found in " + path


def test_research_data_is_kept(tmp_path):
    run_once(tmp_path)
    originals = {}
    for path in glob.glob(os.path.join(INPUT, "*.dcm")):
        ds = pydicom.dcmread(path)
        originals[ds.SeriesDescription.split(" for ")[0]] = ds
    for path in glob.glob(os.path.join(str(tmp_path), "*", "*", "*.dcm")):
        ds = pydicom.dcmread(path)
        assert ds.SliceThickness == 2.5
        assert ds.PatientSex in ["F", "M"] and ds.PatientAge in ["045Y", "060Y"]
        assert ds.Manufacturer == "ACME MEDICAL"
        assert ds.pixel_array.shape == (16, 16)
        assert ds.SeriesDescription in originals          # descriptions cleaned, not deleted


def test_same_input_gives_identical_output(tmp_path):
    run_once(tmp_path / "first")
    run_once(tmp_path / "second")
    first = output_hashes(tmp_path / "first")
    second = output_hashes(tmp_path / "second")
    assert len(first) == 3
    assert first == second


def test_resume_after_a_stopped_run(tmp_path):
    run_once(tmp_path / "full")
    stopped = tmp_path / "stopped"
    run_once(stopped)
    # pretend the run was killed: one output file is gone and the audit log ends with half a line
    audit_path = os.path.join(str(stopped), "audit.jsonl")
    with open(audit_path) as f:
        lines = f.readlines()
    removed = json.loads(lines[2])["output"]
    os.remove(removed)
    with open(audit_path, "w") as f:
        f.write(lines[0] + lines[1] + lines[2][:40])
    summary = run_once(stopped)
    assert summary["lineage lines"] == 3
    assert output_hashes(stopped) == output_hashes(tmp_path / "full")


def test_lineage_is_complete_and_correct(tmp_path):
    run_once(tmp_path)
    with open(os.path.join(str(tmp_path), "lineage.jsonl")) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 3
    for line in lines:
        with open(line["output"], "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == line["output_sha256"]
        with open(line["input"], "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == line["input_sha256"]
        assert len(line["policy_sha256"]) == 64
        assert line["audit_line"] >= 1


def test_the_same_patient_gets_the_same_pseudonym_and_shift(tmp_path):
    run_once(tmp_path)
    patients = set()
    dates = {}
    for path in glob.glob(os.path.join(str(tmp_path), "*", "*", "*.dcm")):
        ds = pydicom.dcmread(path)
        patients.add(ds.PatientID)
        dates.setdefault(ds.PatientID, set()).add(ds.StudyDate)
    assert len(patients) == 2
    for patient, study_dates in dates.items():
        assert len(study_dates) == 1          # both files of patient A are shifted by the same days
