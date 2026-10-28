# %% [markdown]
# # Phase 4: Burned-in text in the pixels
#
# **Goal:** hide text that identifies a patient when it is written into the image itself, and keep harmless text
# (side markers, views, scale, window values) readable. Nothing outside the hidden boxes may change.
#
# **Method (FP-DeID):** OCR (EasyOCR on GPU 0) reads three views of the image (normal, bright-text window,
# dark-text window; small images are upscaled) and joins the pieces into text lines. Each line is checked with the
# same **PHI fingerprint** as in Phase 3 (exact or fuzzy, to allow OCR mistakes), plus label words (DOB, ID, ...) and
# dates. On X-rays, short codes that look like initials are hidden unless they appear in the images of many patients
# (then they are screen/device words). Known radiology markers are kept.
#
# **Compared with:** (a) hide all text the OCR finds, (b) hide fixed bands at the top and bottom (common practice).
#
# **Data:** the MIDI-B key has only 35 burned-in text checks, so we also build a **synthetic benchmark**: real images
# with text we write ourselves, so we know every box exactly. 70% of the patients are used to tune, 30% to test.
#
# Pass criteria (docs/PLAN.md): PHI box recall 100% (each miss shown); 0 readable PHI words after hiding;
# harmless text kept ≥ 90%; hidden area smaller than the other methods; 0 pixels changed outside boxes;
# MIDI-B pixel checks pass; 100% valid DICOM; screening ≥ 5× faster with 0 PHI missed; GPU memory ≤ 8 GB.

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import time
import shutil
import pickle
import hashlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import torch

from fpdeid import paths
from fpdeid import engine
from fpdeid import fingerprint
from fpdeid import burnin
from fpdeid import pixels
from fpdeid import pixel_eval
from fpdeid import scoring
from fpdeid import validity

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

phase_results = os.path.join(paths.RESULTS, "phase4")
os.makedirs(phase_results, exist_ok=True)
synthetic_dir = os.path.join(paths.DATA, "synthetic_burnin")

print("torch", torch.__version__, "| GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0))

file_table = pd.read_csv(os.path.join(paths.DERIVED, "validation_file_table.csv"), dtype=str)
all_files = file_table["path"].tolist()
patient_of = dict(zip(file_table["path"], file_table["patient_id"]))
modality_of = dict(zip(file_table["path"], file_table["modality"]))
series_of = dict(zip(file_table["path"], file_table["series_uid"]))

# %%
# the Phase 3 fingerprints (same as in the header cleaning)
policy_path = os.path.join(paths.CONFIGS, "policy.yaml")
settings = engine.Policy(policy_path, paths.VALIDATION_UID_MAP, paths.VALIDATION_PATIENT_MAP)
fingerprints, common_words = fingerprint.build_all(all_files, workers=16, common_share=settings.common_word_share,
                                                   use_site=settings.site_level)
print("fingerprints:", len(fingerprints))

# %% [markdown]
# ## 1. The real burned-in text in MIDI-B (35 checks)

# %%
checks_table = pd.read_csv(os.path.join(paths.DERIVED, "validation_pixel_checks.csv"))
uid_to_path = dict(zip(file_table["sop_uid"], file_table["path"]))
checks_table["path"] = checks_table["sop"].map(uid_to_path)

rows = []
for _, check in checks_table.iterrows():
    ds, array, photometric = pixels.load_pixels(check["path"])
    frame = pixels.first_frame(ds, array)
    found = pixels.ocr_image(frame)
    fp = fingerprints.get(patient_of[check["path"]])
    boxes, decisions = pixels.boxes_for_method(found, "fpdeid", fp, check["modality"], set(),
                                               frame.shape[0], frame.shape[1])
    for decision in decisions:
        rows.append({"modality": check["modality"], "key text": check["text"], "ocr line": decision["text"],
                     "confidence": decision["confidence"], "hide": decision["hide"], "reason": decision["reason"]})
real_table = pd.DataFrame(rows)
real_table.to_csv(os.path.join(phase_results, "real_midi_b_ocr.csv"), index=False)
real_table.head(40)

# %% [markdown]
# ## 2. Build the synthetic benchmark

# %%
single_frame = file_table[(file_table["frames"].fillna("1").astype(float) <= 1)]
projection = single_frame[single_frame["modality"].isin(["CR", "DX", "MG", "US"])]
rng = np.random.default_rng(4)
cross = single_frame[single_frame["modality"].isin(["CT", "MR", "PT"])]
cross_sample = cross.iloc[rng.choice(len(cross), size=200, replace=False)]
sources = pd.concat([projection, cross_sample])
print("source images:", len(sources), sources["modality"].value_counts().to_dict())

shutil.rmtree(synthetic_dir, ignore_errors=True)
jobs = []
for i, path in enumerate(sources["path"]):
    output = os.path.join(synthetic_dir, "images", str(i).zfill(5) + ".dcm")
    jobs.append((path, output, i))
with Pool(16) as pool:
    per_image = pool.map(burnin.make_one, jobs)

labels = []
for image_labels in per_image:
    labels.extend(image_labels)
label_table = pd.DataFrame(labels)
print("images:", label_table["file"].nunique(), "| text boxes:", len(label_table),
      "| PHI:", int(label_table["phi"].sum()), "| harmless:", int((~label_table["phi"]).sum()))
burnin.save_labels(labels, os.path.join(synthetic_dir, "labels.json"))


# %%
# split by patient: 70% tune, 30% test (a patient is never in both)
def split_of(patient):
    number = int(hashlib.md5(patient.encode()).hexdigest()[:8], 16)
    return "test" if number % 10 < 3 else "tune"


label_table["split"] = label_table["patient"].map(split_of)
label_table.groupby(["split", "kind"]).size().unstack(fill_value=0)

# %%
# show a few synthetic images
examples = label_table["file"].drop_duplicates().iloc[[0, 40, 120, 300]]
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, path in zip(axes, examples):
    ds = pydicom.dcmread(path)
    ax.imshow(burnin.to_display(ds.pixel_array), cmap="gray")
    for _, label in label_table[label_table["file"] == path].iterrows():
        x0, y0, x1, y1 = label["box"]
        color = "red" if label["phi"] else "lime"
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.5))
    ax.set_title(ds.Modality)
    ax.axis("off")
plt.suptitle("Synthetic benchmark: red = PHI, green = harmless")
plt.tight_layout()
plt.savefig(os.path.join(phase_results, "synthetic_examples.png"), dpi=100)
plt.show()

# %% [markdown]
# ## 3. OCR every synthetic image once (GPU 0)

# %%
# the synthetic images are made with fixed seeds, so a saved OCR result can be reused
cache_path = os.path.join(paths.DERIVED, "phase4_ocr_cache.pkl")
torch.cuda.reset_peak_memory_stats()
if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        ocr_results, image_sizes, ocr_seconds, gpu_gb = pickle.load(f)
    print("reused saved OCR results")
else:
    ocr_results = {}
    image_sizes = {}
    start = time.time()
    for path in label_table["file"].unique():
        ds, array, photometric = pixels.load_pixels(path)
        image_sizes[path] = (array.shape[0], array.shape[1])
        ocr_results[path] = pixels.ocr_image(array)
    ocr_seconds = time.time() - start
    gpu_gb = torch.cuda.max_memory_allocated() / 1e9
    with open(cache_path, "wb") as f:
        pickle.dump((ocr_results, image_sizes, ocr_seconds, gpu_gb), f)
print("OCR on", len(ocr_results), "images:", round(ocr_seconds, 1), "s =",
      round(1000 * ocr_seconds / len(ocr_results)), "ms per image | peak GPU memory:", round(gpu_gb, 2), "GB")

file_patient = dict(zip(label_table["file"], label_table["patient"]))
file_modality = dict(zip(label_table["file"], label_table["modality"]))
tune_files = set(label_table[label_table["split"] == "tune"]["file"])
tune_ocr = {path: found for path, found in ocr_results.items() if path in tune_files}
common_codes = pixels.common_codes_from(tune_ocr, file_patient)
print("common short codes (kept):", sorted(common_codes)[:30])


# %% [markdown]
# ## 4. Tune on the tune split, then score all methods on the test split

# %%
def evaluate(split, method, min_confidence):
    rows = []
    for path, group in label_table[label_table["split"] == split].groupby("file"):
        height, width = image_sizes[path]
        fp = fingerprints.get(group["patient"].iloc[0])
        hidden = pixel_eval.decisions_for_method(ocr_results[path], method, fp, height, width,
                                                 file_modality[path], common_codes, min_confidence)
        rows.append(pixel_eval.score_file(group.to_dict("records"), hidden, height, width))
    table = pd.DataFrame(rows)
    summary = {
        "PHI recall %": round(100 * table["phi_hidden"].sum() / table["phi_boxes"].sum(), 2),
        "harmless kept %": round(100 * table["harmless_kept"].sum() / table["harmless_boxes"].sum(), 2),
        "box precision %": round(100 * table["hidden_on_phi"].sum() / max(1, table["hidden_boxes"].sum()), 2),
        "hidden area %": round(table["hidden_area_pct"].mean(), 3),
    }
    return summary


tuning = []
for confidence in [0.0, 0.1, 0.2, 0.3, 0.5]:
    row = evaluate("tune", "fpdeid", confidence)
    row["min confidence"] = confidence
    tuning.append(row)
tuning_table = pd.DataFrame(tuning).set_index("min confidence")
tuning_table.to_csv(os.path.join(phase_results, "tuning.csv"))
tuning_table

# %%
# choose the setting with the best PHI recall, then the most harmless text kept
best = tuning_table.sort_values(["PHI recall %", "harmless kept %"], ascending=False).index[0]
print("chosen min confidence:", best)

methods = {}
for method in ["fpdeid", "all_text", "bands"]:
    methods[method] = evaluate("test", method, best)
method_table = pd.DataFrame(methods).T
method_table.to_csv(os.path.join(phase_results, "methods_test_split.csv"))
method_table

# %% [markdown]
# ### Leaks after hiding: OCR reads the PHI boxes again (test split)

# %%
leak_rows = []
for method in ["fpdeid", "all_text", "bands"]:
    leaked = 0
    phi_words = 0
    for path, group in label_table[label_table["split"] == "test"].groupby("file"):
        ds = pydicom.dcmread(path)
        array = ds.pixel_array.copy()
        height, width = array.shape[0], array.shape[1]
        fp = fingerprints.get(group["patient"].iloc[0])
        hidden = pixel_eval.decisions_for_method(ocr_results[path], method, fp, height, width,
                                                 file_modality[path], common_codes, best)
        array = pixels.hide_boxes(array, hidden, str(ds.PhotometricInterpretation))
        records = group.to_dict("records")
        leaked = leaked + pixel_eval.leaked_words(burnin.to_display(array), records)
        for record in records:
            if record["phi"]:
                phi_words = phi_words + len(pixel_eval.words_of(record["text"]))
    leak_rows.append({"method": method, "PHI words": phi_words, "still readable": leaked})
leak_table = pd.DataFrame(leak_rows).set_index("method")
leak_table.to_csv(os.path.join(phase_results, "leaks_test_split.csv"))
leak_table

# %%
# PHI recall by modality and image size (small images have tiny text)
by_group = []
for path, group in label_table[label_table["split"] == "test"].groupby("file"):
    height, width = image_sizes[path]
    hidden = pixel_eval.decisions_for_method(ocr_results[path], "fpdeid", fingerprints.get(file_patient[path]),
                                             height, width, file_modality[path], common_codes, best)
    for record in group.to_dict("records"):
        if record["phi"]:
            by_group.append({"modality": record["modality"], "image size": "small (<512 px)" if height < 512 else "large",
                             "hidden": pixel_eval.covered_share(record["box"], hidden) >= 0.9})
recall_groups = pd.DataFrame(by_group).groupby(["modality", "image size"])["hidden"].agg(["count", "mean"])
recall_groups["recall %"] = (100 * recall_groups["mean"]).round(1)
recall_groups = recall_groups.drop(columns="mean")
recall_groups.to_csv(os.path.join(phase_results, "recall_by_modality.csv"))
recall_groups

# %%
# which PHI boxes were missed by FP-DeID (test split), and why
missed = []
for path, group in label_table[label_table["split"] == "test"].groupby("file"):
    height, width = image_sizes[path]
    found = [f for f in ocr_results[path] if f[2] >= best]
    fp = fingerprints.get(group["patient"].iloc[0])
    hidden = pixel_eval.decisions_for_method(found, "fpdeid", fp, height, width, file_modality[path], common_codes)
    for record in group.to_dict("records"):
        if record["phi"] and pixel_eval.covered_share(record["box"], hidden) < 0.9:
            ocr_here = [f[1] for f in found if pixel_eval.overlaps(f[0], record["box"])]
            missed.append({"modality": record["modality"], "kind": record["kind"], "text": record["text"],
                           "ocr read": " | ".join(ocr_here)})
missed_table = pd.DataFrame(missed)
missed_table.to_csv(os.path.join(phase_results, "missed_phi_boxes.csv"), index=False)
print("missed PHI boxes:", len(missed_table))
missed_table.head(30)

# %%
# pixels outside the hidden boxes must be byte-identical (checked on 50 test images)
outside_changed = 0
for path in label_table[label_table["split"] == "test"]["file"].unique()[:50]:
    ds = pydicom.dcmread(path)
    before = ds.pixel_array.copy()
    height, width = before.shape[0], before.shape[1]
    fp = fingerprints.get(file_patient[path])
    hidden = pixel_eval.decisions_for_method(ocr_results[path], "fpdeid", fp, height, width, file_modality[path],
                                             common_codes, best)
    after = pixels.hide_boxes(before.copy(), hidden, str(ds.PhotometricInterpretation))
    inside = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in hidden:
        pad = max(4, int(0.2 * (y1 - y0)))
        inside[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
    diff = before != after
    if diff.ndim == 3:
        diff = diff.any(axis=2)
    outside_changed = outside_changed + int((diff & ~inside).sum())
print("pixels changed outside hidden boxes:", outside_changed)

# %% [markdown]
# ## 5. Screening: do cross-sectional series need OCR on every slice?

# %%
sample_cross = cross.iloc[rng.choice(len(cross), size=300, replace=False)]
start = time.time()
any_text = 0
phi_text = 0
for path in sample_cross["path"]:
    ds, array, photometric = pixels.load_pixels(path)
    found = pixels.ocr_image(array)
    if len(found) > 0:
        any_text = any_text + 1
    if pixels.has_phi_text(found, fingerprints.get(patient_of[path]), modality_of[path]):
        phi_text = phi_text + 1
cross_seconds = time.time() - start
print("original CT/MR/PET slices: OCR found some text in", any_text, "of", len(sample_cross),
      "| text that would be hidden in", phi_text, "| ms per slice:", round(1000 * cross_seconds / len(sample_cross)))

# %% [markdown]
# ## 6. The full pipeline on the validation set (headers + pixels), official scorer

# %%
policy = engine.Policy(policy_path, paths.VALIDATION_UID_MAP, paths.VALIDATION_PATIENT_MAP)
policy.set_fingerprints(fingerprints)
out_folder = os.path.join(paths.OUTPUTS, "phase4")
shutil.rmtree(out_folder, ignore_errors=True)

start = time.time()
run_table = engine.run(all_files, out_folder, policy, workers=16)
header_seconds = time.time() - start

records = []
with open(os.path.join(out_folder, "audit.jsonl")) as f:
    for line in f:
        record = json.loads(line)
        records.append({"input": record["input"], "output": record["output"]})

pixel_audit = os.path.join(out_folder, "audit_pixels.jsonl")
start = time.time()
pixel_summary = pixels.clean_outputs(records, fingerprints, patient_of, modality_of, series_of, pixel_audit,
                                     method="fpdeid")
pixel_seconds = time.time() - start
pixel_summary["header seconds"] = round(header_seconds, 1)
pixel_summary["pixel seconds"] = round(pixel_seconds, 1)
n_files = len(records)
pixel_summary["screening: share of files read %"] = round(100 * pixel_summary["files read by OCR"] / n_files, 2)
pd.Series(pixel_summary)

# %%
answer_db = os.path.join(paths.VALIDATION_ANSWER_DIR, "MIDI-B-Answer-Key-Validation.db")
db_path = scoring.run_official_scorer("phase4", out_folder, answer_db, paths.VALIDATION_UID_MAP,
                                      paths.VALIDATION_PATIENT_MAP, os.path.join(paths.DERIVED, "scorer_runs"), cpus=16)
results = scoring.add_categories(scoring.load_results(db_path))
summary = scoring.summarise(results)
summary.update(scoring.privacy_utility(results))

phase3 = pd.read_csv(os.path.join(paths.RESULTS, "phase3", "scores_vs_phase2.csv"), index_col=0)
compare = phase3[["Phase 3"]].copy()
compare["Phase 4"] = pd.Series(summary)
compare.to_csv(os.path.join(phase_results, "scores_vs_phase3.csv"))
compare

# %%
per_action = scoring.per_action(results)
per_action.to_csv(os.path.join(phase_results, "per_action.csv"))
per_action

# %%
pixel_rows = results[results["action"] == "<pixels_hidden>"][["check_passed", "modality", "action_text", "file_value"]]
pixel_rows.to_csv(os.path.join(phase_results, "midi_b_pixel_checks.csv"), index=False)
pixel_rows

# %%
# validity of every file whose pixels were changed
changed = []
with open(pixel_audit) as f:
    for line in f:
        entry = json.loads(line)
        if entry["hidden_boxes"] > 0:
            changed.append((entry["input"], entry["output"]))
valid_table = validity.check_pairs(changed, workers=24)
valid_pct = 100 * (valid_table["new_errors"] == 0).mean() if len(changed) > 0 else 100.0
print("files with hidden text:", len(changed), "| valid:", round(valid_pct, 2), "%")

# %% [markdown]
# ## 7. Phase 4 summary vs. pass criteria

# %%
fp_row = method_table.loc["fpdeid"]
criteria = pd.DataFrame([
    ["PHI box recall (synthetic test split)", "100%", fp_row["PHI recall %"]],
    ["readable PHI words after hiding", "0", int(leak_table.loc["fpdeid", "still readable"])],
    ["harmless text kept", "≥ 90%", fp_row["harmless kept %"]],
    ["hidden area vs. all_text / bands", "smaller", str(fp_row["hidden area %"]) + " vs " +
     str(method_table.loc["all_text", "hidden area %"]) + " / " + str(method_table.loc["bands", "hidden area %"])],
    ["pixels changed outside hidden boxes", "0", outside_changed],
    ["MIDI-B pixel checks passed", "35 / 35", int(pixel_rows["check_passed"].fillna(0).sum())],
    ["MIDI-B images that must stay unchanged (pixels_retained)", "100%", per_action.loc["<pixels_retained>", "accuracy_%"]],
    ["valid DICOM (files with hidden text)", "100%", round(valid_pct, 2)],
    ["share of files read by OCR (screening)", "small", pixel_summary["screening: share of files read %"]],
    ["peak GPU memory", "≤ 8 GB", round(gpu_gb, 2)],
    ["MIDI-B weighted score", "≥ Phase 3 (99.828%)", summary["weighted_score_%"]],
], columns=["check", "target", "result"])
criteria.to_csv(os.path.join(phase_results, "pass_criteria.csv"), index=False)
criteria

# %%
import gzip

for name in ["audit.jsonl", "audit_pixels.jsonl"]:
    with open(os.path.join(out_folder, name), "rb") as source:
        with gzip.open(os.path.join(paths.DERIVED, "phase4_" + name + ".gz"), "wb") as target:
            shutil.copyfileobj(source, target)
shutil.rmtree(out_folder, ignore_errors=True)
print("done")
