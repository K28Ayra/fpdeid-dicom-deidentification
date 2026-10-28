"""Phase 5: test the auditor with leaks that we plant ourselves (so we know the right answer).

We take clean output files, copy them, and put ONE real identifier of that patient back into each copy,
in a different place and a different written form each time. A good auditor must find all of them.
The values are read straight from the original file here (not with the auditor's code).
"""

import os
import random

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from fpdeid import burnin

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

PLACES = ["ImageComments", "SeriesDescription (appended)", "private LO tag", "private bytes (UN)",
          "inside a sequence", "PatientComments"]


def write_date(date8, style):
    year = date8[0:4]
    month = date8[4:6]
    day = date8[6:8]
    if style == 0:
        return day + "." + month + "." + year
    if style == 1:
        return month + "/" + day + "/" + year
    if style == 2:
        return year + "-" + month + "-" + day
    if style == 3:
        return day + " " + MONTHS[int(month) - 1] + " " + year
    return date8


def secrets_of(original_path):
    """The identifiers of one original file that we can plant: [(kind, text)]."""
    ds = pydicom.dcmread(original_path, stop_before_pixels=True)
    found = []
    name = str(ds.get("PatientName", ""))
    parts = [p for p in name.split("^") if p.strip() != ""]
    if len(parts) >= 2:
        found.append(("name", parts[1] + " " + parts[0]))        # FIRST LAST (other order than in the header)
        found.append(("name", parts[0]))                          # surname alone
    if len(str(ds.get("PatientID", ""))) >= 4:
        found.append(("id", "MRN " + str(ds.PatientID)))
    if len(str(ds.get("AccessionNumber", ""))) >= 4:
        found.append(("id", str(ds.AccessionNumber)))
    birth = str(ds.get("PatientBirthDate", ""))
    if len(birth) == 8:
        found.append(("date", "born " + write_date(birth, random.randint(0, 3))))
    study = str(ds.get("StudyDate", ""))
    if len(study) == 8:
        found.append(("date", "scan of " + write_date(study, random.randint(0, 3))))
    doctor = str(ds.get("ReferringPhysicianName", ""))
    if "^" in doctor:
        found.append(("name", "Dr " + doctor.replace("^", " ")))
    institution = str(ds.get("InstitutionName", ""))
    if len(institution) >= 4:
        found.append(("place or contact", institution))
    found.append(("uid", str(ds.StudyInstanceUID)))
    return found


def plant_header(output_path, original_path, copy_path, plant_number):
    """Copy output_path to copy_path with one identifier planted. Returns a description of the plant."""
    ds = pydicom.dcmread(output_path)
    choices = secrets_of(original_path)
    kind, text = choices[plant_number % len(choices)]
    place = PLACES[plant_number % len(PLACES)]

    if kind == "uid":
        # an original UID put back into a reference (as a careless tool might do)
        item = Dataset()
        item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        item.ReferencedSOPInstanceUID = text
        ds.ReferencedStudySequence = Sequence([item])
        place = "UID inside a sequence"
    elif place == "ImageComments":
        ds.ImageComments = text
    elif place == "SeriesDescription (appended)":
        ds.SeriesDescription = str(ds.get("SeriesDescription", "")) + " " + text
    elif place == "private LO tag":
        block = ds.private_block(0x0013, "PLANTED TEST", create=True)
        block.add_new(0x01, "LO", text[:64])
    elif place == "private bytes (UN)":
        block = ds.private_block(0x0015, "PLANTED BYTES", create=True)
        block.add_new(0x01, "UN", ("info=" + text).encode("latin-1"))
    elif place == "inside a sequence":
        item = Dataset()
        item.RequestedProcedureDescription = text[:64]
        ds.RequestAttributesSequence = Sequence([item])
    else:
        ds.PatientComments = text

    os.makedirs(os.path.dirname(copy_path), exist_ok=True)
    ds.save_as(copy_path, enforce_file_format=False)
    return {"copy": copy_path, "kind": kind, "text": text, "place": place, "where": "header"}


def plant_pixels(output_path, original_path, copy_path, plant_number):
    """Copy an output image and write the patient's name or ID into a corner of the pixels."""
    import cv2

    ds = pydicom.dcmread(output_path)
    original = pydicom.dcmread(original_path, stop_before_pixels=True)
    name = str(original.get("PatientName", "")).replace("^", ", ")
    if plant_number % 2 == 0 and len(name) > 3:
        kind = "name"
        text = name
    else:
        kind = "id"
        text = "ID " + str(original.PatientID)

    array = ds.pixel_array.copy()
    frames = int(ds.get("NumberOfFrames", 1) or 1)
    frame = array[0] if frames > 1 else array
    height = frame.shape[0]
    width = frame.shape[1]
    scale = max(0.4, height / 900)
    thickness = max(1, int(scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    corner = plant_number % 4
    x = 10 if corner in (0, 2) else int(width * 0.45)
    y = 10 if corner in (0, 1) else height - text_height - baseline - 10
    # written with the brightest value, like real burned-in text (works for 8-bit, 16-bit and colour images)
    burnin.draw_text(frame, text, x, y, scale, frame.max())
    if frames > 1:
        array[0] = frame
    else:
        array = frame

    ds.PixelData = array.astype(ds.pixel_array.dtype).tobytes()
    if "YBR" in str(ds.PhotometricInterpretation):
        ds.PhotometricInterpretation = "RGB"
    from pydicom.uid import ExplicitVRLittleEndian
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    os.makedirs(os.path.dirname(copy_path), exist_ok=True)
    ds.save_as(copy_path, enforce_file_format=False)
    return {"copy": copy_path, "kind": kind, "text": text, "place": "pixels (corner " + str(corner) + ")",
            "where": "pixels"}
