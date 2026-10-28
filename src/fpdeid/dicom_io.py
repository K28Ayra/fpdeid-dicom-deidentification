"""Small helpers to find and read DICOM files."""

import os
from multiprocessing import Pool

import pandas as pd
import pydicom


def list_files(folder):
    """Return every file path below a folder (sorted, so runs are repeatable)."""
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            full_path = os.path.join(root, name)
            paths.append(full_path)
    paths.sort()
    return paths


def read_summary(path):
    """Read only the header of one file and return a small summary dict."""
    summary = {"path": path, "readable": False, "error": ""}
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as error:
        summary["error"] = str(error)[:200]
        return summary

    summary["readable"] = True
    summary["size_bytes"] = os.path.getsize(path)
    summary["patient_id"] = str(ds.get("PatientID", ""))
    summary["study_uid"] = str(ds.get("StudyInstanceUID", ""))
    summary["series_uid"] = str(ds.get("SeriesInstanceUID", ""))
    summary["sop_uid"] = str(ds.get("SOPInstanceUID", ""))
    summary["modality"] = str(ds.get("Modality", ""))
    summary["sop_class"] = str(ds.get("SOPClassUID", ""))

    transfer_syntax = ""
    if hasattr(ds, "file_meta"):
        transfer_syntax = str(ds.file_meta.get("TransferSyntaxUID", ""))
    summary["transfer_syntax"] = transfer_syntax

    summary["rows"] = ds.get("Rows", None)
    summary["columns"] = ds.get("Columns", None)
    summary["frames"] = ds.get("NumberOfFrames", 1)
    summary["burned_in_annotation"] = str(ds.get("BurnedInAnnotation", ""))
    summary["n_private_tags"] = count_private(ds)
    return summary


def count_private(ds):
    count = 0
    for element in ds.iterall():
        if element.tag.is_private:
            count = count + 1
    return count


def index_folder(folder, workers=8):
    """Read the header summary of every file in a folder, in parallel."""
    paths = list_files(folder)
    with Pool(workers) as pool:
        summaries = pool.map(read_summary, paths, chunksize=50)
    table = pd.DataFrame(summaries)
    return table
