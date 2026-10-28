# Phase 1: Data exploration and baselines

**Notebook:** [`notebooks/01_exploration_and_baseline.ipynb`](../notebooks/01_exploration_and_baseline.ipynb)
**Code:** [`src/fpdeid/baselines.py`](../src/fpdeid/baselines.py), [`src/fpdeid/ps315.py`](../src/fpdeid/ps315.py),
[`src/fpdeid/census.py`](../src/fpdeid/census.py), [`src/fpdeid/phi_locations.py`](../src/fpdeid/phi_locations.py),
[`src/fpdeid/scoring.py`](../src/fpdeid/scoring.py)
**Results:** [`results/phase1/`](../results/phase1/)

## What this phase is about

Before building my own method, I want to answer two questions:

1. **Where does patient information hide in these files?**
2. **How good is the standard way of cleaning them?** This gives me a number that every later phase must beat.

A DICOM file has a **header** (a long list of labelled fields, called *tags*, like `PatientName` or `StudyDate`)
and the **pixels** (the image itself). Patient information (PHI) can be in the header, inside free-text fields, inside
vendor-specific *private* tags, or written into the pixels.

## Part 1, What is in the headers?

I opened every file and listed every tag path it contains (including tags nested inside sequences).

| | Count |
|---|---|
| Distinct tag paths | 1,444 |
| Standard tags (top level) | 439 |
| Standard tags inside sequences | 384 |
| Private tags (top level) | 602 |
| Private tags inside sequences | 19 |
| Vendor groups ("private creators") | 23 (mostly GE PET/CT) |

Check from the plan: **98.7%** of the tag paths that the answer key checks are present in my census (1,277 of 1,294).
The 17 missing ones are a few private tags nested inside private sequences and one retired tag.

## Part 2, Where does PHI hide? (the main finding)

For every file I took the patient's own name, ID, birth date and accession number from their usual ("home") fields,
and searched for them in **all other fields** of the same file.

> **In 98.8% of the files (23,641 of 23,921), the patient's own name or ID also appears in some other field.**

![Fields that contain the patient's name or ID](../results/phase1/phi_outside_home_tag.png)

| Where the name / ID was found | Files | What the standard profile does with that field |
|---|---|---|
| Series Description (e.g. *"PET-CT SERIES for Peggy Peters"*) | 9,236 | deletes it |
| Additional Patient History | 7,004 | deletes it |
| Protocol Name | 6,975 | replaces it with a dummy |
| Study Comments | 5,815 | deletes it |
| Requested Procedure Description | 5,655 | empties it |
| Study Description | 5,168 | deletes it |
| Private tags | 1,098 | deletes all private tags |

This shows the core problem of rule-per-field cleaning: these descriptions are **useful for research**
(they say what the scan is), but they also contain the name. A fixed rule can only *keep* them (leak) or *delete* them
(lose useful data). This is the motivation for my **PHI fingerprint** idea (Phase 3): find the patient's own values and
remove only those words, keeping the rest of the description.

Two more findings:

- **Series Number contains dates** in many files (for example `20160807`). A number field that a standard tool always keeps
  can still leak a date. This causes 21,268 failed checks in the best baseline.
- **Burned-in text:** the answer key has 35 pixel checks, all in X-ray images (DX, CR, MG), with text like
  *"JAMES RUSSELL [M] 12.13.2018 / DOB: 10.21.1947"* written into the image. No file flags this, so it must be found by
  looking at the pixels (Phase 4).

## Part 3, The answer key

| Check type | Count | Meaning |
|---|---|---|
| `text_retained` | 3,453,539 | this value must stay |
| `tag_retained` | 1,101,091 | this field must still exist |
| `text_notnull` | 618,539 | this field must not be empty |
| `text_removed` | 343,583 | this value must be gone (PHI) |
| `uid_changed` | 234,418 | this UID must be replaced |
| `uid_consistent` | 234,418 | ...with the expected new UID |
| `date_shifted` | 139,774 | this date must be changed |
| `patid_consistent` | 23,921 | Patient ID must be the expected pseudonym |
| `pixels_retained` | 23,886 | the image must not change |
| `pixels_hidden` | 35 | burned-in text must be hidden |

So most checks are about **keeping useful data**, not only removing PHI. A tool that deletes too much scores badly.

The official score is weighted: **70% HIPAA** (removing identifiers), **20% DICOM standard** (valid, consistent files),
**10% TCIA best practice** (keeping useful data).

## Part 4, Baselines

I built three baselines, from "do nothing" to "what a careful standard tool does":

| Name | What it does |
|---|---|
| **B0, IDs only** | Replaces only UIDs and the Patient ID (so the scorer can find the files). Everything else is kept. This is the floor. |
| **BA, Basic Profile** | The DICOM PS3.15 *Basic Application Level Confidentiality Profile*: the official rule table, all private tags removed, no date shifting. |
| **BB, Basic + options** | Basic Profile + *Retain Longitudinal Temporal Information (shifted dates)* + *Retain Patient Characteristics* + *Retain Device Identity*. |

Scores on the full validation set (23,921 files, official MIDI-B script):

| | B0 | BA | **BB (best)** |
|---|---|---|---|
| **Weighted MIDI-B score** | 61.51% | 95.31% | **95.87%** |
| HIPAA checks | 45.83% | 99.99% | 99.99% |
| DICOM checks | 99.89% | 99.89% | 99.89% |
| TCIA best-practice checks | 94.49% | 53.42% | 59.01% |
| Privacy checks failed (leaks) | 454,324 | 21,358 | 31,131 |
| Utility checks failed (over-cleaning) | 55,371 | 1,670,457 | 1,458,139 |
| Valid DICOM (1,000-file sample, no new errors) | 100% | 99.8% | 99.5% |
| Speed | 1,846 files/s | 1,765 files/s | 1,463 files/s |

![Baseline scores](../results/phase1/baseline_scores.png)

**What this tells me:**
- The standard profile is very good at privacy (99.99% of HIPAA checks), but it **throws away about a third of the
  useful data**. The biggest losses are private PET/vendor tags (for example 79,548 failures on *Private Creator* and about
  8,000 on each GE PET acquisition parameter) and Study/Series Descriptions (about 23,000 each).
- The remaining leaks are mostly **dates hidden in Series Number** and names inside free text.
- So the room for improvement is in **utility** (keep more), while keeping privacy at the top.

Per modality (share of all checks passed, BB): structured reports (99.7%) and mammography (99.4%) are almost perfect,
while MRI (68.7%), PET (75.6%) and CT (83.0%) lose the most, because they have the most private tags and descriptions.

## Problems I solved

**1. Getting the official rules without typing them by hand.**
The DICOM standard (PS3.15, Table E.1-1) lists 657 attributes and the action for each one. I parse the table directly from the
official NEMA XML file ([`scripts/parse_ps315_table.py`](../scripts/parse_ps315_table.py)) into
[`configs/ps315_table_e1_1.csv`](../configs/ps315_table_e1_1.csv), so the baseline follows the standard exactly.

**2. Combined actions like "X/Z" or "Z/D".**
The table sometimes says "remove *or* empty, depending on the file type". I chose the option that keeps the file valid:
keep the field (empty or dummy) instead of deleting it, except for sequences, where an empty sequence is not allowed.

**3. The Patient ID must not simply be emptied.**
The Basic Profile empties Patient ID, but research data (and the scorer) need a consistent pseudonym.
So Patient ID is always replaced with the pseudonym from the mapping table.

**4. Rules on sequences mean "work inside the sequence".**
For sequences like *Referenced Image Sequence* the table says "U" (new UIDs). My first version tried to put a UID into the
sequence itself, which crashed on 561 CT files. Now, for a sequence marked U, C or D, the tool keeps the sequence and applies
the rules to the fields inside it. A unit test checks this.

**5. Keeping files valid.**
My first version made only about 70% of files valid, because "dummy" sequences were left with an empty item, and some
"remove or empty" sequences were emptied where the standard needs at least one item. After fixing both, 99.5–99.8% of files
have no new error. The small rest comes from one sequence (*Acquisition Context Sequence*) that must be present even when empty;
I handle this in Phase 2 with type-aware rules.

**6. Never touching the image by accident.**
The scorer checks that the image data is byte-for-byte unchanged (MD5 hash). My tool only edits header fields and writes
the pixel data back exactly as it was read, in the original transfer syntax. All 23,886 `pixels_retained` checks pass.

**7. A 2.8 GB answer key.**
I read it in chunks of 2,000 rows to keep memory low, and I wrote a helper that makes a small answer key for just the
dev-mini patients, so quick experiments take minutes instead of the full run.

**8. Speed.**
Files are processed in parallel (16 processes). Each process loads the mapping tables once instead of once per file.
The whole validation set (23,921 files) is de-identified in about 15 seconds; scoring takes about 15 minutes.

## How to reproduce

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploration_and_baseline.ipynb
python -m pytest -q tests
```
