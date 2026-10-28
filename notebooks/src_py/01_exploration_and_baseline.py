# %% [markdown]
# # Phase 1: Data exploration and baselines
#
# **Goal:** understand where patient information (PHI) hides in the MIDI-B validation data,
# and measure what standard de-identification gets. Every later phase must beat these numbers.
#
# Parts:
# 1. Tag census: which tags exist (standard, private, inside sequences).
# 2. Where does PHI appear outside its "home" tag? (motivation for the PHI fingerprint)
# 3. Free text and burned-in text.
# 4. The answer key: which actions are required.
# 5. Three baselines, scored with the official MIDI-B script:
#    - **B0 ids-only**: replace UIDs and Patient ID only (floor)
#    - **BA basic**: DICOM PS3.15 Basic Profile
#    - **BB basic+**: Basic Profile + Retain Dates (shifted) + Retain Patient Characteristics + Retain Device Identity
# 6. Error analysis of the best baseline.

# %%
import os
import glob
import shutil
import sqlite3
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

from fpdeid import paths
from fpdeid import census
from fpdeid import phi_locations
from fpdeid import baselines
from fpdeid import scoring
from fpdeid import validity

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

phase_results = os.path.join(paths.RESULTS, "phase1")
os.makedirs(phase_results, exist_ok=True)

file_table_path = os.path.join(paths.DERIVED, "validation_file_table.csv")
file_table = pd.read_csv(file_table_path, dtype=str)
all_files = file_table["path"].tolist()
print("files:", len(all_files))

# %% [markdown]
# ## 1. Tag census
#
# We open every file (header only) and record every tag path, e.g. `(0008,1140)>(0008,1155)` for a tag inside a sequence.

# %%
start = time.time()
tag_census = census.build_census(all_files, workers=16)
print("seconds:", round(time.time() - start, 1))
print("distinct tag paths:", len(tag_census))

tag_census.to_csv(os.path.join(paths.DERIVED, "tag_census_with_examples.csv"), index=False)
no_examples = tag_census.drop(columns=["examples"])
no_examples.to_csv(os.path.join(phase_results, "tag_census.csv"), index=False)

# %%
is_private = tag_census["private"] == True
is_nested = tag_census["depth"] > 0

census_summary = pd.Series({
    "distinct tag paths": len(tag_census),
    "standard, top level": int((~is_private & ~is_nested).sum()),
    "standard, inside sequences": int((~is_private & is_nested).sum()),
    "private, top level": int((is_private & ~is_nested).sum()),
    "private, inside sequences": int((is_private & is_nested).sum()),
})
census_summary

# %%
# which private creators are there (vendors)?
private_rows = tag_census[is_private].copy()
creator = private_rows["path"].str.extract(r'"([^"]+)"')[0]
private_rows["creator"] = creator
creator_counts = private_rows.groupby("creator")["path"].count().sort_values(ascending=False)
print("private creators:", len(creator_counts))
creator_counts.head(20)

# %% [markdown]
# ## 2. Where does PHI appear outside its home tag?
#
# For each file we take its Patient Name (parts), Patient ID, Birth Date, Accession Number and Other Patient IDs,
# and look for these values inside every *other* text tag of the same file.

# %%
start = time.time()
phi_hits = phi_locations.find_phi_outside_home(all_files, workers=16)
print("seconds:", round(time.time() - start, 1))

n_files = len(all_files)
n_files_with_hit = phi_hits["file"].nunique()
share = 100 * n_files_with_hit / n_files
print("files where PHI appears outside its home tag:", n_files_with_hit, "of", n_files, "=", round(share, 1), "%")

# %%
where = phi_hits.groupby(["home_tag", "found_in_name"]).agg(
    hits=("file", "count"),
    files=("file", "nunique"),
    private=("private", "max"),
)
where = where.sort_values("files", ascending=False).reset_index()
where.to_csv(os.path.join(phase_results, "phi_outside_home_tag.csv"), index=False)
where.head(25)

# %%
# what would the Basic Profile do with these tags? (X = remove, K/not listed = keep, ...)
rules_basic = baselines.ps315.load_rules([])
rules_clean = baselines.ps315.load_rules(["clean_descriptors"])

keyword_to_action = {}
keyword_to_action_clean = {}
for tag_int in rules_basic:
    name = rules_basic[tag_int][2]
    keyword_to_action[name] = rules_basic[tag_int][0]
    keyword_to_action_clean[name] = rules_clean[tag_int][0]

where["basic_profile"] = where["found_in_name"].map(keyword_to_action).fillna("keep (not listed)")
where["with_clean_descriptors"] = where["found_in_name"].map(keyword_to_action_clean).fillna("keep (not listed)")
where.loc[where["private"] == True, "basic_profile"] = "X (private)"
where.head(25)

# %%
plot_data = phi_hits.groupby("found_in_name")["file"].nunique().sort_values().tail(15)
plt.figure(figsize=(8, 5))
plt.barh(plot_data.index, plot_data.values)
plt.xlabel("number of files")
plt.title("Tags (other than the home tag) that contain the patient's own name / ID")
plt.tight_layout()
plt.savefig(os.path.join(phase_results, "phi_outside_home_tag.png"), dpi=120)
plt.show()

# %% [markdown]
# ## 3. Burned-in text (pixels)

# %%
burned = file_table.groupby("modality")["burned_in_annotation"].value_counts().unstack(fill_value=0)
burned

# %%
# show a few images from modalities that often have text in the pixels (US, SC, DX)
examples = file_table[file_table["modality"].isin(["US", "SC", "OT"])].head(4)
if len(examples) > 0:
    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 4))
    if len(examples) == 1:
        axes = [axes]
    for ax, path in zip(axes, examples["path"]):
        ds = pydicom.dcmread(path)
        image = ds.pixel_array
        if image.ndim == 4 or (image.ndim == 3 and image.shape[-1] != 3):
            image = image[0]
        ax.imshow(image, cmap="gray")
        ax.set_title(str(ds.Modality) + " " + str(ds.get("BurnedInAnnotation", "")))
        ax.axis("off")
    plt.show()

# %% [markdown]
# ## 4. The answer key
#
# Each answer-key row is one instance (or series/study/patient) with a JSON list of checks.
# Check types: `<text_removed>`, `<text_retained>`, `<tag_retained>`, `<text_notnull>`, `<date_shifted>`,
# `<uid_changed>`, `<uid_consistent>`, `<patid_consistent>`, `<pixels_hidden>`, `<pixels_retained>`.

# %%
answer_files = glob.glob(os.path.join(paths.VALIDATION_ANSWER_DIR, "**", "*.db"), recursive=True)
answer_db = answer_files[0]
print("answer key:", answer_db)

# %%
# Open the JSON of every row into one long table of checks (one row per required check).
import json


def strip_brackets(text):
    if text is None:
        return ""
    return str(text).strip("<>")


check_rows = []
connection = sqlite3.connect(answer_db)
query = "SELECT Modality, SOPInstanceUID, AnswerData FROM answer_data"
n_answer_rows = 0
for chunk in pd.read_sql(query, connection, chunksize=2000):
    n_answer_rows = n_answer_rows + len(chunk)
    for _, row in chunk.iterrows():
        answer_dict = json.loads(row["AnswerData"])
        for key in answer_dict:
            check = answer_dict[key]
            check_rows.append({
                "modality": row["Modality"],
                "scope": strip_brackets(check.get("scope")),
                "action": strip_brackets(check.get("action")),
                "tag_name": strip_brackets(check.get("tag_name")),
                "tag": strip_brackets(check.get("tag")),
                "tag_ds": strip_brackets(check.get("tag_ds")),
            })
connection.close()

checks = pd.DataFrame(check_rows)
del check_rows
print("answer rows (instances):", n_answer_rows)
print("required checks:", len(checks))

action_counts = checks["action"].value_counts()
action_counts.to_csv(os.path.join(phase_results, "answer_key_actions.csv"))
action_counts

# %%
top_tags = checks.groupby(["action", "tag_name"]).size().sort_values(ascending=False)
top_tags = top_tags.reset_index(name="checks")
top_tags.to_csv(os.path.join(phase_results, "answer_key_top_tags.csv"), index=False)
top_tags.head(30)

# %%
# Pass criterion: every tag path that the answer key checks must appear in our tag census.
import re


def key_path_to_census_path(key_path):
    path = re.sub(r">?\[<\d+>\]<?", ">", key_path)
    path = path.replace("<", "").replace(">>", ">")
    return path.upper()


key_paths = set()
for tag_ds in checks["tag_ds"].unique():
    if tag_ds != "":
        key_paths.add(key_path_to_census_path(tag_ds))

census_tags = set(tag_census["path"].str.upper())
found = key_paths & census_tags
coverage = 100 * len(found) / len(key_paths)
print("tag paths in answer key:", len(key_paths))
print("found in our census    :", len(found), "(", round(coverage, 2), "% )")
not_found = sorted(key_paths - census_tags)
print("not found:", len(not_found))
for path in not_found[:10]:
    print("  ", path)

# %%
# required checks per modality and action
by_modality = pd.crosstab(checks["modality"], checks["action"], margins=True)
by_modality.to_csv(os.path.join(phase_results, "answer_key_by_modality.csv"))
by_modality

# %%
# the burned-in text checks (pixels_hidden) — how many and in which modalities?
pixel_checks = checks[checks["action"] == "pixels_hidden"]
print("pixels_hidden checks:", len(pixel_checks))
pixel_checks["modality"].value_counts()

# %% [markdown]
# ## 5. Baselines
#
# For each baseline: de-identify all files -> score with the official script -> check validity on a sample
# -> delete the output files (to save disk), keep only the scores.

# %%
mappings = baselines.Mappings(paths.VALIDATION_UID_MAP, paths.VALIDATION_PATIENT_MAP)

scorer_out = os.path.join(paths.DERIVED, "scorer_runs")
os.makedirs(scorer_out, exist_ok=True)

all_results = {}
all_summaries = {}

rng = np.random.default_rng(0)
validity_sample_size = 1000

for name in ["B0", "BA", "BB"]:
    print("=====", name, "=====")

    saved_db = os.path.join(scorer_out, "phase1_" + name, "validation_results.db")
    saved_summary = os.path.join(paths.DERIVED, "summary_" + name + ".csv")
    if os.path.exists(saved_db) and os.path.exists(saved_summary):
        results = scoring.add_categories(scoring.load_results(saved_db))
        all_results[name] = results
        all_summaries[name] = pd.read_csv(saved_summary, index_col=0).iloc[:, 0].to_dict()
        print("reused saved results")
        continue

    out_folder = os.path.join(paths.OUTPUTS, "phase1_" + name)
    shutil.rmtree(out_folder, ignore_errors=True)

    start = time.time()
    run_table = baselines.run_baseline(all_files, out_folder, name, mappings, workers=16)
    deid_seconds = time.time() - start
    n_errors = (run_table["error"] != "").sum()
    print("de-identified in", round(deid_seconds), "s, errors:", n_errors)

    start = time.time()
    db_path = scoring.run_official_scorer("phase1_" + name, out_folder, answer_db,
                                          paths.VALIDATION_UID_MAP, paths.VALIDATION_PATIENT_MAP,
                                          scorer_out, cpus=16)
    print("scored in", round(time.time() - start), "s")
    results = scoring.load_results(db_path)
    results = scoring.add_categories(results)
    all_results[name] = results

    ok_rows = run_table[run_table["error"] == ""]
    sample_index = rng.choice(len(ok_rows), size=min(validity_sample_size, len(ok_rows)), replace=False)
    sample = ok_rows.iloc[sample_index]
    pairs = list(zip(sample["input"], sample["output"]))
    valid_table = validity.check_pairs(pairs, workers=16)
    valid_pct = 100 * (valid_table["new_errors"] == 0).mean()

    summary = scoring.summarise(results)
    summary.update(scoring.privacy_utility(results))
    summary["valid_files_%_sample"] = round(valid_pct, 2)
    summary["deid_errors"] = int(n_errors)
    summary["files_per_second"] = round(len(all_files) / deid_seconds, 1)
    all_summaries[name] = summary
    pd.Series(summary).to_csv(saved_summary)
    valid_table.to_csv(os.path.join(paths.DERIVED, "validity_" + name + ".csv"), index=False)

    shutil.rmtree(out_folder, ignore_errors=True)
    print(pd.Series(summary))

# %%
baseline_table = pd.DataFrame(all_summaries)
baseline_table.to_csv(os.path.join(phase_results, "baseline_scores.csv"))
baseline_table

# %% [markdown]
# ### Accuracy per action type

# %%
per_action_tables = []
for name in all_results:
    table = scoring.per_action(all_results[name])
    table = table[["checks", "accuracy_%"]].rename(columns={"accuracy_%": name})
    per_action_tables.append(table[[name]])
per_action_all = pd.concat(per_action_tables, axis=1)
per_action_all.insert(0, "checks", scoring.per_action(all_results["BA"])["checks"])
per_action_all.to_csv(os.path.join(phase_results, "baseline_per_action.csv"))
per_action_all

# %% [markdown]
# ### Accuracy per modality

# %%
modality_tables = []
for name in all_results:
    table = all_results[name].copy()
    table["passed"] = table["check_passed"].fillna(0).astype(float)
    accuracy = table.groupby("modality")["passed"].mean() * 100
    modality_tables.append(accuracy.round(2).rename(name))
per_modality = pd.concat(modality_tables, axis=1)
per_modality.to_csv(os.path.join(phase_results, "baseline_per_modality.csv"))
per_modality

# %%
plot_table = baseline_table.loc[["weighted_score_%", "privacy_pass_%", "utility_pass_%"]].T
plot_table.plot(kind="bar", figsize=(8, 4), ylim=(0, 100))
plt.ylabel("%")
plt.title("Phase 1 baselines (MIDI-B validation)")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(phase_results, "baseline_scores.png"), dpi=120)
plt.show()

# %% [markdown]
# ## 6. Error analysis of the best baseline

# %%
best = baseline_table.loc["weighted_score_%"].astype(float).idxmax()
print("best baseline:", best)
failures = scoring.top_failures(all_results[best], n=25)
failures.to_csv(os.path.join(phase_results, "best_baseline_top_failures.csv"), index=False)
failures

# %%
# leaks (privacy failures) vs over-cleaning (utility failures), per tag
best_results = all_results[best]
failed = best_results[best_results["check_passed"].fillna(0).astype(float) < 1]
failed_privacy = failed[failed["action"].isin(scoring.PRIVACY_ACTIONS)]
failed_utility = failed[failed["action"].isin(scoring.UTILITY_ACTIONS)]
print("leaks (privacy failures):", len(failed_privacy))
print("over-cleaning (utility failures):", len(failed_utility))

print("\nTop leaking tags:")
print(failed_privacy["tag_name"].value_counts().head(15))
print("\nTop over-cleaned tags:")
print(failed_utility["tag_name"].value_counts().head(15))
