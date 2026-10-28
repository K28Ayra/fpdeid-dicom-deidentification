"""Small tests for the Phase 1 baselines. They use a tiny DICOM file made in code (no real data)."""

import os

import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from fpdeid import baselines


def make_file(folder):
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
    ds.StudyDate = "20200110"
    ds.Modality = "CT"
    ds.SliceThickness = "2.5"
    ds.add_new(0x00091010, "LO", "private value")
    ds.private_block(0x0009, "TEST CREATOR", create=True)

    path = os.path.join(folder, "in.dcm")
    ds.save_as(path, enforce_file_format=True)
    return path


def make_mappings(folder):
    uid_csv = os.path.join(folder, "uids.csv")
    patient_csv = os.path.join(folder, "patients.csv")
    uid_table = pd.DataFrame({
        "id_old": ["1.2.3.4.1", "1.2.3.4.2", "1.2.3.4.3"],
        "id_new": ["9.9.1", "9.9.2", "9.9.3"],
    })
    uid_table.to_csv(uid_csv, index=False)
    pd.DataFrame({"id_old": ["12345"], "id_new": ["MIDI_TEST_001"]}).to_csv(patient_csv, index=False)
    return baselines.Mappings(uid_csv, patient_csv)


def run_one(tmp_path, name):
    input_path = make_file(str(tmp_path))
    mappings = make_mappings(str(tmp_path))
    out_root = os.path.join(str(tmp_path), "out_" + name)
    table = baselines.run_baseline([input_path], out_root, name, mappings, workers=1)
    assert table["error"].iloc[0] == ""
    return pydicom.dcmread(table["output"].iloc[0])


def test_b0_changes_only_ids(tmp_path):
    ds = run_one(tmp_path, "B0")
    assert ds.SOPInstanceUID == "9.9.3"
    assert ds.PatientID == "MIDI_TEST_001"
    assert str(ds.PatientName) == "DOE^JOHN"      # B0 keeps everything else


def test_basic_profile_removes_phi(tmp_path):
    ds = run_one(tmp_path, "BA")
    assert ds.PatientID == "MIDI_TEST_001"
    assert str(ds.get("PatientName", "")) == ""
    assert str(ds.get("PatientBirthDate", "")) == ""
    assert ds.StudyInstanceUID == "9.9.1"
    assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID
    assert ds.PatientIdentityRemoved == "YES"
    private_tags = [e for e in ds if e.tag.is_private]
    assert len(private_tags) == 0
    assert ds.SliceThickness == 2.5                # acquisition data kept


def test_basic_plus_shifts_dates(tmp_path):
    ds = run_one(tmp_path, "BB")
    assert ds.StudyDate != "20200110"
    assert len(ds.StudyDate) == 8


def test_date_shift_is_same_for_same_patient(tmp_path):
    mappings = make_mappings(str(tmp_path))
    first = mappings.date_offset_days("12345")
    second = mappings.date_offset_days("12345")
    assert first == second
    assert -3650 <= first <= -365


def test_uid_inside_sequence_is_replaced(tmp_path):
    """Referenced Image Sequence has action U in PS3.15: the UIDs inside must change, the sequence stays."""
    from pydicom.dataset import Dataset as Item
    input_path = make_file(str(tmp_path))
    ds = pydicom.dcmread(input_path)
    item = Item()
    item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    item.ReferencedSOPInstanceUID = "1.2.3.4.3"
    ds.ReferencedImageSequence = [item]
    ds.save_as(input_path, enforce_file_format=True)

    mappings = make_mappings(str(tmp_path))
    out_root = os.path.join(str(tmp_path), "out")
    table = baselines.run_baseline([input_path], out_root, "BA", mappings, workers=1)
    assert table["error"].iloc[0] == ""
    out = pydicom.dcmread(table["output"].iloc[0])
    assert out.ReferencedImageSequence[0].ReferencedSOPInstanceUID == "9.9.3"
