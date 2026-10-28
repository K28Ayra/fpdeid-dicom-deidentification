"""Tests for the Phase 2 rules engine (frozen Phase 2 policy), on a tiny DICOM file made in code."""

import json
import os

import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from fpdeid import engine, paths


def make_file(folder):
    os.makedirs(folder, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = "1.2.3.4.3"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = "1.2.3.4.3"
    ds.StudyInstanceUID = "1.2.3.4.1"
    ds.SeriesInstanceUID = "1.2.3.4.2"
    ds.PatientName = "DOE^JOHN"
    ds.PatientID = "12345"
    ds.PatientBirthDate = "19650312"
    ds.PatientSex = "M"
    ds.StudyDate = "20200110"
    ds.SeriesDate = "20200115"
    ds.SeriesNumber = "20200110"           # a date hidden in a number field
    ds.SeriesDescription = "CT CHEST for John Doe"
    ds.Modality = "CT"
    ds.SliceThickness = "2.5"
    ds.Manufacturer = "GE MEDICAL SYSTEMS"

    # a reference to another image, inside a sequence
    item = Dataset()
    item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    item.ReferencedSOPInstanceUID = "1.2.3.4.3"
    ds.ReferencedImageSequence = [item]

    # private tags: one number (keep), one text (remove)
    block = ds.private_block(0x0009, "TEST CREATOR", create=True)
    block.add_new(0x01, "DS", "3.75")
    block.add_new(0x02, "LO", "John Doe home address")

    path = os.path.join(folder, "in.dcm")
    ds.save_as(path, enforce_file_format=True)
    return path


def make_policy(folder):
    uid_csv = os.path.join(folder, "uids.csv")
    patient_csv = os.path.join(folder, "patients.csv")
    uid_table = pd.DataFrame({
        "id_old": ["1.2.3.4.1", "1.2.3.4.2", "1.2.3.4.3"],
        "id_new": ["9.9.1", "9.9.2", "9.9.3"],
    })
    uid_table.to_csv(uid_csv, index=False)
    pd.DataFrame({"id_old": ["12345"], "id_new": ["MIDI_TEST_001"]}).to_csv(patient_csv, index=False)
    policy_path = os.path.join(paths.CONFIGS, "policy_phase2.yaml")
    return engine.Policy(policy_path, uid_csv, patient_csv)


def run_engine(tmp_path):
    input_path = make_file(str(tmp_path))
    policy = make_policy(str(tmp_path))
    out_root = os.path.join(str(tmp_path), "out")
    table = engine.run([input_path], out_root, policy, workers=1)
    assert table["error"].iloc[0] == ""
    ds = pydicom.dcmread(table["output"].iloc[0])
    with open(os.path.join(out_root, "audit.jsonl")) as f:
        record = json.loads(f.readline())
    return ds, record, input_path


def test_identifiers_are_removed_or_pseudonymised(tmp_path):
    ds, record, _ = run_engine(tmp_path)
    assert ds.PatientID == "MIDI_TEST_001"
    assert "DOE" not in str(ds.get("PatientName", ""))
    assert str(ds.get("PatientBirthDate", "")) == ""
    assert ds.SOPInstanceUID == "9.9.3"
    assert ds.file_meta.MediaStorageSOPInstanceUID == "9.9.3"


def test_dates_are_shifted_and_intervals_kept(tmp_path):
    ds, record, input_path = run_engine(tmp_path)
    original = pydicom.dcmread(input_path)
    assert ds.StudyDate != original.StudyDate
    old_gap = int(original.SeriesDate) - int(original.StudyDate)
    new_gap = int(ds.SeriesDate) - int(ds.StudyDate)
    # same month in both -> the day difference is the same number
    assert old_gap == new_gap


def test_number_that_looks_like_a_date_is_shifted(tmp_path):
    ds, record, _ = run_engine(tmp_path)
    assert str(ds.SeriesNumber) != "20200110"


def test_research_fields_are_kept(tmp_path):
    ds, record, _ = run_engine(tmp_path)
    assert ds.SliceThickness == 2.5
    assert ds.Manufacturer == "GE MEDICAL SYSTEMS"
    assert ds.PatientSex == "M"


def test_uid_inside_sequence_is_pseudonymised(tmp_path):
    ds, record, _ = run_engine(tmp_path)
    assert ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID == "9.9.3"


def test_private_numbers_kept_private_text_removed(tmp_path):
    ds, record, _ = run_engine(tmp_path)
    block = ds.private_block(0x0009, "TEST CREATOR")
    assert float(block[0x01].value) == 3.75
    assert 0x01 in block
    assert 0x02 not in block


def test_every_change_is_in_the_audit_log(tmp_path):
    ds, record, _ = run_engine(tmp_path)
    logged = set(change["keyword"] for change in record["changes"])
    for keyword in ["PatientName", "PatientID", "PatientBirthDate", "StudyDate", "SeriesNumber", "SOPInstanceUID"]:
        assert keyword in logged
    for change in record["changes"]:
        assert change["gdpr"] != ""
        assert change["ps315_option"] != ""
        assert "DOE" not in change["new_value"]


def test_same_input_gives_same_output(tmp_path):
    first, _, _ = run_engine(tmp_path / "a")
    second, _, _ = run_engine(tmp_path / "b")
    assert first.StudyDate == second.StudyDate
    assert first.SeriesNumber == second.SeriesNumber


class FakeElement:
    """Just the parts of a DICOM element that the time-stamp check looks at."""

    def __init__(self, name, vr, value):
        self.name = name
        self.VR = vr
        self.value = value
        self.VM = 1


def test_unix_time_stamp_is_shifted_but_counts_are_not():
    stamp = FakeElement("[Actual series data time stamp]", "SL", 1228050572)
    count = FakeElement("[PET delayed_events]", "SL", 1228050572)
    assert engine.looks_like_timestamp(stamp, stamp.value) is True
    assert engine.looks_like_timestamp(count, count.value) is False
    assert engine.shift_timestamp(stamp, -10) == 1228050572 - 10 * 86400
