"""Tests for the evaluation tools (Phases 1-5) on the tiny synthetic files: no GPU, no OCR model, no MIDI-B data.

The pixel step is tested with a ready-made OCR result (the OCR cache), so the decisions and the hiding are tested
without the OCR model.
"""

import glob
import json
import os
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pydicom
import pytest

from fpdeid import auditor
from fpdeid import burnin
from fpdeid import census
from fpdeid import checks
from fpdeid import dicom_io
from fpdeid import fingerprint
from fpdeid import paths
from fpdeid import phi_locations
from fpdeid import pixel_eval
from fpdeid import pixels
from fpdeid import plant_leaks
from fpdeid import reid_attack
from fpdeid import run
from fpdeid import scoring
from fpdeid import validity

INPUT = os.path.join(os.path.dirname(__file__), "data", "input")


@pytest.fixture(scope="module")
def cleaned(tmp_path_factory):
    """One pipeline run on the 3 synthetic files. Returns (output folder, [(input, output)])."""
    folder = tmp_path_factory.mktemp("cleaned")
    run.run(INPUT, str(folder), os.path.join(paths.CONFIGS, "policy.yaml"), workers=2, do_pixels=False)
    pairs = []
    with open(os.path.join(str(folder), "audit.jsonl")) as f:
        for line in f:
            record = json.loads(line)
            pairs.append((record["input"], record["output"]))
    return str(folder), sorted(pairs)


def make_xray(folder, name="xray.dcm"):
    """A 256 x 256 X-ray made from a synthetic test file."""
    ds = pydicom.dcmread(os.path.join(INPUT, "a1.dcm"))
    ds.Modality = "DX"
    ds.Rows = 256
    ds.Columns = 256
    image = np.tile(np.arange(256, dtype=np.uint16), (256, 1)) * 4
    ds.PixelData = image.tobytes()
    path = os.path.join(str(folder), name)
    ds.save_as(path, enforce_file_format=True)
    return path


# ---------------------------------------------------------------
# Phase 1 helpers: file index, tag census, PHI outside its home tag
# ---------------------------------------------------------------

def test_file_index_and_census():
    files = dicom_io.list_files(INPUT)
    assert len(files) == 3
    summary = dicom_io.read_summary(files[0])
    assert summary["modality"] == "CT"
    table = census.build_census(files, workers=2)
    assert len(table) > 30


def test_phi_is_found_outside_its_home_tag():
    found = phi_locations.find_phi_outside_home(dicom_io.list_files(INPUT), workers=2)
    text = " ".join(str(value) for value in found.to_numpy().ravel())
    assert "Series Description" in text          # "CHEST CT for Alice Tester"
    assert "Image Comments" in text              # "MRN A1234567 ..." and "Bob Sample, call ..."


# ---------------------------------------------------------------
# Phase 2 checks: audit coverage, utility, dates, UIDs
# ---------------------------------------------------------------

def test_independent_checks_on_a_cleaned_file(cleaned):
    folder, pairs = cleaned
    audited = {}
    with open(os.path.join(folder, "audit.jsonl")) as f:
        for line in f:
            record = json.loads(line)
            audited[record["input"]] = set(change["tag"] for change in record["changes"])
    input_uids = set()
    for original, output in pairs:
        ds = pydicom.dcmread(original)
        input_uids.update([ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID])
    jobs = [(original, output, audited[original]) for original, output in pairs]
    rows = checks.run_checks(jobs, input_uids, workers=2)
    for row in rows:
        assert row["changed_not_logged"] == 0
        assert row["utility_kept"] == row["utility_present"]
        assert row["date_pairs_same"] == row["date_pairs"]
        assert row["original_uids_left"] == 0
        assert row["invalid_uids"] == 0


def test_validity_tool_or_skip(cleaned):
    if not os.path.exists(validity.DCIODVFY):
        pytest.skip("dciodvfy is not installed here")
    folder, pairs = cleaned
    table = validity.check_pairs(pairs, workers=2)
    assert (table["new_errors"] == 0).all()


# ---------------------------------------------------------------
# Scoring helpers (with a tiny table in the scorer's format)
# ---------------------------------------------------------------

def fake_results():
    rows = [
        {"action": "<text_removed>", "check_passed": 1, "hipaa_m": "x", "hipaa_z": "", "tcia_p15": "", "tcia_ptkb": "",
         "tcia_rev": "", "tag_name": "<Patient's Name>", "tag_ds": "<(0010,0010)>", "file_value": "<>"},
        {"action": "<text_removed>", "check_passed": 0, "hipaa_m": "x", "hipaa_z": "", "tcia_p15": "", "tcia_ptkb": "",
         "tcia_rev": "", "tag_name": "<Image Comments>", "tag_ds": "<(0020,4000)>", "file_value": "<call 5551234567>"},
        {"action": "<text_retained>", "check_passed": 0, "hipaa_m": "", "hipaa_z": "", "tcia_p15": "x", "tcia_ptkb": "",
         "tcia_rev": "", "tag_name": "<Series Description>", "tag_ds": "<(0008,103E)>", "file_value": "<CHEST>"},
        {"action": "<tag_retained>", "check_passed": 1, "hipaa_m": "", "hipaa_z": "", "tcia_p15": "", "tcia_ptkb": "",
         "tcia_rev": "", "tag_name": "<Rows>", "tag_ds": "<(0028,0010)>", "file_value": "<16>"},
        {"action": "<uid_consistent>", "check_passed": 1, "hipaa_m": "", "hipaa_z": "", "tcia_p15": "", "tcia_ptkb": "",
         "tcia_rev": "", "tag_name": "<SOP Instance UID>", "tag_ds": "<(0008,0018)>", "file_value": "<1.2>"},
    ]
    return pd.DataFrame(rows)


def test_scoring_summary_and_causes():
    table = scoring.add_categories(fake_results())
    assert list(table["score_cat"]) == ["hipaa", "hipaa", "tcia", "dicom", "dicom"]
    summary = scoring.summarise(table)
    assert summary["hipaa_accuracy_%"] == 50.0
    groups = scoring.privacy_utility(table)
    assert groups["privacy_failed"] == 1 and groups["utility_failed"] == 1
    assert scoring.per_action(table).loc["<text_removed>", "failed"] == 1
    assert len(scoring.top_failures(table)) == 2
    failed = scoring.failures_by_cause(table)
    causes = list(failed["cause"])
    assert causes[0].startswith("LEAK")
    assert causes[1] == "over-cleaning in free text"


def test_answer_key_subset(tmp_path):
    source = os.path.join(str(tmp_path), "key.db")
    connection = sqlite3.connect(source)
    pd.DataFrame({"SOPInstanceUID": ["1", "2", "3"], "AnswerData": ["{}", "{}", "{}"]}).to_sql(
        "answer_data", connection, index=False)
    connection.close()
    subset = scoring.make_answer_subset(source, ["2"], os.path.join(str(tmp_path), "subset.db"))
    connection = sqlite3.connect(subset)
    rows = connection.execute("SELECT SOPInstanceUID FROM answer_data").fetchall()
    connection.close()
    assert rows == [("2",)]


# ---------------------------------------------------------------
# Phase 4: pixels without OCR (the OCR result is given), and the benchmark tools
# ---------------------------------------------------------------

def test_pixel_step_hides_only_the_identifying_line(tmp_path):
    original = make_xray(tmp_path, "original.dcm")
    output = os.path.join(str(tmp_path), "output.dcm")
    shutil.copy(original, output)
    fingerprints, common = fingerprint.build_all([original], workers=1)
    cache = os.path.join(str(tmp_path), "ocr_cache.jsonl")
    found = [[[10, 10, 120, 30], "TESTER ALICE", 0.95], [[200, 220, 215, 240], "L", 0.99]]
    with open(cache, "w") as f:
        f.write(json.dumps({"output": output, "found": found}) + "\n")
    records = [{"input": original, "output": output}]
    audit = os.path.join(str(tmp_path), "audit_pixels.jsonl")
    summary = pixels.clean_outputs(records, fingerprints, {original: "A1234567"}, {original: "DX"},
                                   {original: "s1"}, audit, ocr_cache_path=cache)
    assert summary["files with hidden text"] == 1
    before = pydicom.dcmread(original).pixel_array
    after = pydicom.dcmread(output).pixel_array
    assert (after[12:28, 12:118] == before.min()).all()          # the name line is hidden
    assert (after[220:240, 200:215] == before[220:240, 200:215]).all()   # the side marker is kept
    with open(audit) as f:
        entry = json.loads(f.readline())
    assert entry["hidden_boxes"] == 1


def test_pixel_helpers(tmp_path):
    path = make_xray(tmp_path)
    ds, array, photometric = pixels.load_pixels(path)
    assert photometric == "MONOCHROME2" and pixels.has_pixels(path)
    assert pixels.first_frame(ds, array).shape == (256, 256)
    views = pixels.make_views(array)
    assert len(views) >= 2 and views[0].dtype == np.uint8
    assert len(pixels.band_boxes(256, 256)) == 2
    records = [{"input": "a", "output": path}, {"input": "b", "output": path}]
    always, samples = pixels.screening_plan(records, {"a": "DX", "b": "CT"}, {"a": "s1", "b": "s2"})
    assert len(always) == 1 and "s2" in samples
    codes = pixels.common_codes_from({"x": [([0, 0, 5, 5], "ABC", 0.9)], "y": [([0, 0, 5, 5], "ABC", 0.9)],
                                      "z": [([0, 0, 5, 5], "ABC", 0.9)]}, {"x": "p1", "y": "p2", "z": "p3"})
    assert "ABC" in codes


def test_benchmark_scoring():
    labels = [{"box": [10, 10, 50, 20], "phi": True, "text": "TESTER ALICE"},
              {"box": [100, 100, 110, 110], "phi": False, "text": "L"}]
    result = pixel_eval.score_file(labels, [[10, 10, 50, 20]], 200, 200)
    assert result["phi_hidden"] == 1 and result["harmless_kept"] == 1 and result["hidden_on_phi"] == 1
    assert pixel_eval.covered_share([10, 10, 50, 20], []) == 0
    assert pixel_eval.words_of("ID: 12 ABC") == ["ABC"]


def test_synthetic_burned_in_text_is_written_with_labels(tmp_path):
    source = make_xray(tmp_path)
    target = os.path.join(str(tmp_path), "burned.dcm")
    labels = burnin.make_one((source, target, 3))
    assert len(labels) > 0
    assert any(label["phi"] for label in labels)
    burned = pydicom.dcmread(target).pixel_array
    original = pydicom.dcmread(source).pixel_array
    assert (burned != original).any()
    burnin.save_labels(labels, os.path.join(str(tmp_path), "labels.json"))


# ---------------------------------------------------------------
# Phase 5: planted leaks, auditor, closed loop, re-identification tables
# ---------------------------------------------------------------

def test_auditor_finds_planted_leaks_and_fixes_them(cleaned, tmp_path):
    folder, pairs = cleaned
    originals = [original for original, output in pairs]
    jobs = []
    plants = []
    for number in range(12):
        original, output = pairs[number % len(pairs)]
        copy = os.path.join(str(tmp_path), str(number) + ".dcm")
        plants.append(plant_leaks.plant_header(output, original, copy, number))
        same_patient = [o for o in originals if pydicom.dcmread(o).PatientID == pydicom.dcmread(original).PatientID]
        jobs.append((same_patient, [copy]))
    hits, elements = auditor.audit_headers(jobs, workers=2)
    found = set(hit["file"] for hit in hits)
    assert all(plant["copy"] in found for plant in plants)
    auditor.fix_header_hits(hits, os.path.join(str(tmp_path), "fixes.jsonl"))
    hits_after, elements = auditor.audit_headers(jobs, workers=2)
    assert hits_after == []


def test_auditor_finds_nothing_in_clean_output(cleaned):
    folder, pairs = cleaned
    by_patient = {}
    for original, output in pairs:
        patient = pydicom.dcmread(original).PatientID
        by_patient.setdefault(patient, ([], []))
        by_patient[patient][0].append(original)
        by_patient[patient][1].append(output)
    hits, elements = auditor.audit_headers(list(by_patient.values()), workers=2)
    assert hits == [] and elements > 100


def test_planted_pixel_text_changes_the_image(cleaned, tmp_path):
    folder, pairs = cleaned
    original = make_xray(tmp_path, "original.dcm")
    output = make_xray(tmp_path, "output.dcm")
    copy = os.path.join(str(tmp_path), "planted.dcm")
    plant = plant_leaks.plant_pixels(output, original, copy, 0)
    assert plant["where"] == "pixels"
    assert (pydicom.dcmread(copy).pixel_array != pydicom.dcmread(output).pixel_array).any()
    assert len(auditor.pixel_views(pydicom.dcmread(copy).pixel_array)) == 2


def test_reidentification_tables(cleaned):
    folder, pairs = cleaned
    originals = reid_attack.patient_table([original for original, output in pairs], workers=2)
    outputs = reid_attack.patient_table([output for original, output in pairs], workers=2)
    assert len(originals) == 2 and len(outputs) == 2
    link = {}
    for original, output in pairs:
        link[pydicom.dcmread(output).PatientID] = pydicom.dcmread(original).PatientID
    result = reid_attack.linkability(outputs, originals, link, ["sex", "age", "weight"])
    assert result["top-1 %"] == 100.0                      # 2 patients with different sex: always linked
    assert reid_attack.information_kept(originals, outputs, link, ["sex", "age", "weight", "study_year"]) > 80
    originals["birth_year"] = originals["birth_year"].astype(str)
    facts = reid_attack.inference(outputs, originals, link)
    assert list(facts["hidden fact"]) == ["study year", "birth year", "hospital"]
    anonymised, suppressed = reid_attack.anonymise(outputs, k=2)
    assert len(anonymised) <= 2


# ---------------------------------------------------------------
# The split runner and the one-command run WITH the pixel step (the OCR is replaced by a stub)
# ---------------------------------------------------------------

def tiny_split(tmp_path):
    """A file table in the same format as the MIDI-B tables, for the 3 synthetic files."""
    rows = []
    for path in sorted(glob.glob(os.path.join(INPUT, "*.dcm"))):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        rows.append({"path": os.path.abspath(path), "readable": "True", "patient_id": ds.PatientID,
                     "modality": ds.Modality, "series_uid": ds.SeriesInstanceUID})
    table_path = os.path.join(str(tmp_path), "file_table.csv")
    pd.DataFrame(rows).to_csv(table_path, index=False)
    return {"file_table": table_path, "uid_map": None, "patient_map": None, "answer_db": None}


def test_split_runner_and_closed_loop(tmp_path, monkeypatch):
    from fpdeid import pipeline
    monkeypatch.setitem(pipeline.SPLITS, "tiny", tiny_split(tmp_path))
    monkeypatch.setattr(pixels, "ocr_file", lambda path: [])
    out = os.path.join(str(tmp_path), "out")
    summary = pipeline.run_split("tiny", out, workers=2)
    assert summary["files"] == 3 and summary["header errors"] == 0
    assert summary["files read by OCR"] == 3                 # CT/MR series: first, middle and last slice
    result = pipeline.audit_and_fix("tiny", out, workers=2)
    assert result["auditor hits after fix"] == 0


def test_one_command_run_with_pixel_step(tmp_path, monkeypatch):
    monkeypatch.setattr(pixels, "ocr_file", lambda path: [])
    summary = run.run(INPUT, str(tmp_path), os.path.join(paths.CONFIGS, "policy.yaml"), workers=2, do_pixels=True)
    assert summary["files read by OCR"] == 3 and summary["files with hidden text"] == 0
    assert os.path.exists(os.path.join(str(tmp_path), "ocr_cache.jsonl"))
    assert summary["lineage lines"] == 3


def test_ner_comparison_cleaner():
    pytest.importorskip("spacy")
    from fpdeid import ner_baseline
    cleaner = ner_baseline.NerCleaner()
    cleaned = cleaner.clean("CT CHEST for John Smith at Boston General Hospital")
    assert "John Smith" not in cleaned
