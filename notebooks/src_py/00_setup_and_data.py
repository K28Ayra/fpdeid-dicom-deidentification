# %% [markdown]
# # Phase 0: Setup and data check
#
# **Goal:** make sure the environment works and the MIDI-B validation data is complete and readable.
#
# What this notebook does:
# 1. Shows the package versions (for reproducibility).
# 2. Compares what TCIA says it has with what we downloaded.
# 3. Opens every DICOM file once (header only) to check it is readable.
# 4. Checks the mapping files (old ID -> new ID) that the scorer needs.
# 5. Checks the answer key (if it is already downloaded).
# 6. Picks a small "dev-mini" subset (20 patients) for fast experiments.
#
# Pass criteria (from `docs/PLAN.md`): 100% of files present, 100% readable,
# 100% of answer-key rows linked to a file, scorer runs.

# %%
import os
import sys
import json
import glob
import sqlite3
import platform

import numpy as np
import pandas as pd
import pydicom

from fpdeid import paths
from fpdeid import dicom_io

os.makedirs(paths.DERIVED, exist_ok=True)
phase_results = os.path.join(paths.RESULTS, "phase0")
os.makedirs(phase_results, exist_ok=True)

# %% [markdown]
# ## 1. Environment

# %%
print("python  :", platform.python_version())
print("pydicom :", pydicom.__version__)
print("pandas  :", pd.__version__)
print("numpy   :", np.__version__)
print("CPUs    :", os.cpu_count())

# %% [markdown]
# ## 2. What TCIA says vs what we downloaded

# %%
with open(paths.VALIDATION_SERIES_LIST) as f:
    series_list = json.load(f)

tcia = pd.DataFrame(series_list)
tcia["ImageCount"] = tcia["ImageCount"].astype(int)
tcia["FileSize"] = tcia["FileSize"].astype(float)

print("series  :", len(tcia))
print("patients:", tcia["PatientID"].nunique())
print("studies :", tcia["StudyInstanceUID"].nunique())
print("images  :", tcia["ImageCount"].sum())
total_gb = tcia["FileSize"].sum() / 1e9
print("size GB :", round(total_gb, 2))

# %%
# count .dcm files that are really on disk, per series folder
on_disk = []
for series in series_list:
    folder = os.path.join(paths.VALIDATION_IMAGES, series["PatientID"], series["SeriesInstanceUID"])
    pattern = os.path.join(folder, "*.dcm")
    n_files = len(glob.glob(pattern))
    on_disk.append(n_files)

tcia["files_on_disk"] = on_disk
tcia["complete"] = tcia["files_on_disk"] == tcia["ImageCount"]

expected = tcia["ImageCount"].sum()
found = tcia["files_on_disk"].sum()
print("files expected:", expected)
print("files found   :", found)
print("completeness  :", round(100 * found / expected, 2), "%")
print("incomplete series:", (~tcia["complete"]).sum())

# %% [markdown]
# ### Inventory per modality

# %%
inventory = tcia.groupby("Modality").agg(
    patients=("PatientID", "nunique"),
    studies=("StudyInstanceUID", "nunique"),
    series=("SeriesInstanceUID", "nunique"),
    images=("ImageCount", "sum"),
    size_mb=("FileSize", "sum"),
)
inventory["size_mb"] = (inventory["size_mb"] / 1e6).round(1)
inventory = inventory.sort_values("images", ascending=False)

total_row = inventory.sum(numeric_only=True)
total_row["patients"] = tcia["PatientID"].nunique()
total_row["studies"] = tcia["StudyInstanceUID"].nunique()
inventory.loc["TOTAL"] = total_row
inventory.to_csv(os.path.join(phase_results, "inventory.csv"))
inventory

# %% [markdown]
# ## 3. Open every file (header only) to check it is readable
#
# We read files in parallel (16 processes). Only the header is read, so memory use stays small.

# %%
file_table = dicom_io.index_folder(paths.VALIDATION_IMAGES, workers=16)

is_dicom_name = file_table["path"].str.endswith(".dcm")
file_table = file_table[is_dicom_name].reset_index(drop=True)

n_files = len(file_table)
n_readable = file_table["readable"].sum()
print("files checked:", n_files)
print("readable     :", n_readable, "(", round(100 * n_readable / n_files, 2), "% )")

unreadable = file_table[~file_table["readable"]]
unreadable[["path", "error"]].head(20)

# %%
# save the full file table (has synthetic IDs, so it stays out of git)
file_table_path = os.path.join(paths.DERIVED, "validation_file_table.csv")
file_table.to_csv(file_table_path, index=False)
print("saved", file_table_path)

# %% [markdown]
# ### Transfer syntaxes, sizes and other quick facts

# %%
syntax_names = {
    "1.2.840.10008.1.2": "Implicit VR Little Endian",
    "1.2.840.10008.1.2.1": "Explicit VR Little Endian",
    "1.2.840.10008.1.2.2": "Explicit VR Big Endian",
    "1.2.840.10008.1.2.4.50": "JPEG Baseline",
    "1.2.840.10008.1.2.4.70": "JPEG Lossless",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 Lossless",
    "1.2.840.10008.1.2.4.91": "JPEG 2000",
    "1.2.840.10008.1.2.5": "RLE Lossless",
}
file_table["syntax_name"] = file_table["transfer_syntax"].map(syntax_names)
file_table["syntax_name"] = file_table["syntax_name"].fillna(file_table["transfer_syntax"])

syntax_counts = file_table["syntax_name"].value_counts()
syntax_counts.to_csv(os.path.join(phase_results, "transfer_syntaxes.csv"))
syntax_counts

# %%
facts = {}
facts["files"] = n_files
facts["readable_pct"] = round(100 * n_readable / n_files, 2)
facts["files_with_private_tags_pct"] = round(100 * (file_table["n_private_tags"] > 0).mean(), 2)
facts["burned_in_annotation_YES"] = int((file_table["burned_in_annotation"] == "YES").sum())
facts["multi_frame_files"] = int((file_table["frames"].astype(float) > 1).sum())
facts["mean_file_kb"] = round(file_table["size_bytes"].mean() / 1024, 1)
pd.Series(facts)

# %% [markdown]
# ## 4. Mapping files (old ID -> new ID)
#
# The scorer finds the answer for an output file by mapping its **new** UID back to the original one.
# So our pipeline must give each UID exactly the new value in this table, and each patient the new
# Patient ID (e.g. `MIDI_1_1_001`). In a hospital this table would be the pseudonymisation key held by the data custodian.

# %%
patient_map = pd.read_csv(paths.VALIDATION_PATIENT_MAP, dtype=str)
uid_map = pd.read_csv(paths.VALIDATION_UID_MAP, dtype=str)
print("patient map rows:", len(patient_map))
print("uid map rows    :", len(uid_map))

patients_in_data = set(file_table["patient_id"])
patients_in_map = set(patient_map["id_old"])
missing_patients = patients_in_data - patients_in_map
print("patients in data not in map:", len(missing_patients))

uids_in_map = set(uid_map["id_old"])
mapping_checks = {}
for column in ["study_uid", "series_uid", "sop_uid"]:
    uids_in_data = set(file_table[column])
    missing = uids_in_data - uids_in_map
    mapping_checks[column] = {"in_data": len(uids_in_data), "missing_from_map": len(missing)}
pd.DataFrame(mapping_checks)

# %% [markdown]
# ## 5. Answer key
#
# The answer key is a SQLite database with one table `answer_data`. Each row is one DICOM instance
# (or series / study) and holds a JSON list of required checks (e.g. `<text_removed>` for a tag).

# %%
answer_files = glob.glob(os.path.join(paths.VALIDATION_ANSWER_DIR, "**", "*.db"), recursive=True)
print("answer key files:", answer_files)

if len(answer_files) > 0:
    connection = sqlite3.connect(answer_files[0])
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", connection)
    print(tables)
    answer_df = pd.read_sql("SELECT * FROM answer_data", connection)
    connection.close()
    print("answer rows:", len(answer_df))
    print("columns    :", list(answer_df.columns))

    sops_in_data = set(file_table["sop_uid"])
    sops_in_key = set(answer_df["SOPInstanceUID"])
    linked = len(sops_in_key & sops_in_data)
    print("answer-key instances linked to a file:", linked, "/", len(sops_in_key))
else:
    print("Answer key not downloaded yet -> see docs/DATA.md")

# %% [markdown]
# ## 6. dev-mini: a small subset for fast experiments
#
# 20 patients, chosen so that every modality appears. Fixed random seed = repeatable.

# %%
rng = np.random.default_rng(42)

patient_modality = tcia.groupby("PatientID")["Modality"].first()
modalities = sorted(patient_modality.unique())

dev_mini = []
per_modality = 20 // len(modalities)
for modality in modalities:
    candidates = patient_modality[patient_modality == modality].index.tolist()
    candidates.sort()
    n_pick = min(per_modality, len(candidates))
    picked = rng.choice(candidates, size=n_pick, replace=False)
    dev_mini.extend(picked.tolist())

others = [p for p in sorted(patient_modality.index) if p not in dev_mini]
n_missing = 20 - len(dev_mini)
extra = rng.choice(others, size=n_missing, replace=False)
dev_mini.extend(extra.tolist())

dev_mini_path = os.path.join(paths.DERIVED, "dev_mini_patients.txt")
with open(dev_mini_path, "w") as f:
    f.write("\n".join(dev_mini))

dev_files = file_table[file_table["patient_id"].isin(dev_mini)]
print("dev-mini patients:", len(dev_mini), "files:", len(dev_files))
dev_files["modality"].value_counts()

# %% [markdown]
# ## Phase 0 summary

# %%
summary = {}
summary["files expected"] = int(expected)
summary["files found"] = int(found)
summary["completeness %"] = round(100 * found / expected, 2)
summary["readable %"] = facts["readable_pct"]
summary["patients missing in patient map"] = len(missing_patients)
summary["answer key present"] = len(answer_files) > 0
summary_table = pd.Series(summary)
summary_table.to_csv(os.path.join(phase_results, "phase0_summary.csv"))
summary_table
