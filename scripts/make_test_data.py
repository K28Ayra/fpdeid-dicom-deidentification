"""Make the 3 tiny synthetic DICOM files used by the integration test (tests/data/).

Usage:  python scripts/make_test_data.py
"""

import os

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

FOLDER = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "input")


def make_file(path, patient, study_uid, series_uid, sop_uid, modality, series_number, description, comment):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2" if modality == "CT" else "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = study_uid + ".9"
    ds.Modality = modality
    ds.PatientName = patient["name"]
    ds.PatientID = patient["id"]
    ds.PatientBirthDate = patient["birth"]
    ds.PatientSex = patient["sex"]
    ds.PatientAge = patient["age"]
    ds.PatientWeight = patient["weight"]
    ds.InstitutionName = "Fairview General Hospital"
    ds.ReferringPhysicianName = "HOUSE^GREGORY"
    ds.AccessionNumber = patient["accession"]
    ds.StudyDate = patient["study_date"]
    ds.SeriesDate = patient["study_date"]
    ds.StudyTime = "101500"
    ds.StudyID = "1"
    ds.SeriesNumber = series_number
    ds.InstanceNumber = 1
    ds.SeriesDescription = description
    ds.ImageComments = comment
    ds.Manufacturer = "ACME MEDICAL"
    ds.ManufacturerModelName = "SCANMASTER 3000"
    ds.SliceThickness = "2.5"
    ds.PixelSpacing = ["0.7", "0.7"]
    block = ds.private_block(0x0019, "ACME PRIVATE", create=True)
    block.add_new(0x10, "LO", "operator note for " + patient["name"].replace("^", " ").title())
    block.add_new(0x11, "DS", "12.5")
    ds.Rows = 16
    ds.Columns = 16
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    image = (np.arange(256, dtype=np.uint16).reshape(16, 16) * 8)
    ds.PixelData = image.tobytes()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.save_as(path, enforce_file_format=True)


def main():
    patient_a = {"name": "TESTER^ALICE", "id": "A1234567", "birth": "19700412", "sex": "F", "age": "045Y",
                 "weight": "61.5", "accession": "ACC0001", "study_date": "20150807"}
    patient_b = {"name": "SAMPLE^BOB", "id": "B7654321", "birth": "19581130", "sex": "M", "age": "060Y",
                 "weight": "82.0", "accession": "ACC0002", "study_date": "20190302"}
    make_file(os.path.join(FOLDER, "a1.dcm"), patient_a, "1.2.826.0.1.111.1", "1.2.826.0.1.111.1.1",
              "1.2.826.0.1.111.1.1.1", "CT", "20150807", "CHEST CT for Alice Tester", "routine scan")
    make_file(os.path.join(FOLDER, "a2.dcm"), patient_a, "1.2.826.0.1.111.1", "1.2.826.0.1.111.1.2",
              "1.2.826.0.1.111.1.2.1", "CT", "2", "CHEST CT LUNG", "MRN A1234567, seen 08/07/2015")
    make_file(os.path.join(FOLDER, "b1.dcm"), patient_b, "1.2.826.0.1.222.1", "1.2.826.0.1.222.1.1",
              "1.2.826.0.1.222.1.1.1", "MR", "3", "BRAIN T1 AXIAL", "Bob Sample, call 555-123-4567")
    print("written to", os.path.abspath(FOLDER))


if __name__ == "__main__":
    main()
