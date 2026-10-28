# FP-DeID, Detailed Phase Plan and Evaluation Criteria

Project: Making hospital scans safe to share (Erasmus Mundus EMIA, Project 2)
Pipeline name: FP-DeID, *Fingerprint-guided, closed-loop De-Identification*

---

## Contents
1. Core idea and novelty
2. Ground rules (data splits, compute, coding style)
3. Global metric definitions (used by all phases)
4. Phases 0–7: tasks, deliverables, metrics, pass criteria
5. Ablation study (how we prove the novelty helps)
6. Final deliverables checklist
7. Risks and fallbacks

---

## 1. Core idea and novelty

### 1.1 What existing methods do
| Approach | Examples | Weakness |
|---|---|---|
| Tag-rule tools | RSNA CTP, pydicom `deid`, dicom-anonymizer | Each tag gets a fixed rule. A name copied into a "keep" field, a free-text comment or a private tag is missed. To stay safe they delete all private tags → over-cleaning. |
| Rules + NER / LLM (most MIDI-B 2024 teams, e.g. the hybrid approach in arXiv 2509.00437) | spaCy / Presidio / transformer NER on free text | NER *guesses* what a name looks like: misses rare or foreign names, flags medical words, needs big models, hard to explain each decision. |
| Pixel text | "black out all text", or OCR + regex | Removes useful text (L/R markers, scale bars, "5 mm"); regex misses names. |
| Evaluation | One score against an answer key | Says how close you are to the curators, not how *re-identifiable* the output still is. |

### 1.2 What FP-DeID adds
1. Patient PHI fingerprint (cross-channel, per patient).
   Collect identifiers from the *structured* fields we are sure about (PatientName, PatientID,
   BirthDate, AccessionNumber, InstitutionName, physician names, addresses, phone numbers, …)
   across *all files of the same patient*. Expand them into variants
   (`DOE^JOHN^A` → "JOHN DOE", "Doe, John", "J. Doe", "JDOE"; `19650312` → "12/03/1965", "1965-03-12",
   "12-Mar-1965", "March 12, 1965"). Search for this fingerprint in every other channel:
   free-text tags, private tags, nested sequences and OCR text from pixels.
   *Claim: the best detector of a patient's leaked identity is the patient's own identity.*
2. Value-aware decisions (less over-cleaning).
   A private tag / text box is removed only if (a) it matches the fingerprint, (b) it matches a
   PHI pattern (date, ID-like, phone, email, address), or (c) the policy says so.
   Numeric acquisition values and harmless text ("L", "AXIAL", "5 mm") are kept.
   Every decision carries a *reason code*.
3. Closed-loop auditor.
   An independent second pass re-reads every output file (headers + OCR) and searches
   for the fingerprint again. Hits are reported as failures, never silently fixed.
4. GDPR made measurable.
   The report turns the Article 29 Working Party anonymisation tests
   (singling out, linkability, inference) into numbers using a re-identification attack,
   and labels each output as *pseudonymised* (still personal data under GDPR Recital 26)
   or closer to *anonymised*.

---

## 2. Ground rules

### 2.1 Data splits (to avoid overfitting to the answer key)
| Split | Source | Use |
|---|---|---|
| dev-mini | 20 subjects sampled from the MIDI-B validation set (stratified by modality) | Fast iteration (< 5 min per run) |
| validation | full MIDI-B synthetic validation set (216 subjects, 23,921 images) | Tuning rules, error analysis |
| test | MIDI-B synthetic test set (322 subjects, 29,660 images) | Run only at the end (Phase 5), once; no rule changes after seeing it |
| synthetic-burn | our own images with text we burn in (known answer) | Pixel phase |
| IXI subset | ~30 T1 head MRI volumes | Optional defacing |

### 2.2 Compute and storage
- Code in the repository; data in `ayra_data/` (git-ignored).
- Process one file at a time (streaming); never load the whole dataset into RAM.
- GPU 0 only (GPU 1 is used by another user). OCR is the only GPU step.
- Large intermediate outputs are deleted after their metrics are saved.

### 2.3 Coding style (as requested)
- One notebook per phase in `notebooks/` (Colab style), simple beginner code.
- Long expressions are split into temporary variables; no long one-liners.
- Functions that work are moved to `src/fpdeid/` so later phases and tests can reuse them.
- Each phase ends with a results table saved in `results/phaseN/` and a stop for review.

---

## 3. Global metric definitions

These are the metrics referred to throughout the plan.

### 3.1 Answer-key metrics (MIDI-B)
The answer key lists, for each file and each tag (and the pixel regions), the required action
(remove, blank, replace/pseudonymise, date-shift, UID change, keep unchanged, redact pixel region).

| Metric | Definition |
|---|---|
| Action accuracy (MIDI-B score) | correct actions / all required actions (official scoring script) |
| Accuracy per action type | same, split by remove / replace / date shift / UID / keep / pixel |
| Accuracy per modality | same, split by CT / MR / PT / CR-DX / US / other |
| Accuracy per tag group | patient, study, equipment, free text, private, sequences |

### 3.2 Privacy metrics (did we remove the PHI?)
Positive class = an element that holds PHI.
| Metric | Definition |
|---|---|
| PHI recall (= 1 − leak rate) | PHI elements correctly handled / all PHI elements |
| Leak count | number of PHI values (from the answer key or the fingerprint) still found in the output, listed one by one |
| Residual fingerprint hits | auditor hits in the output (headers + OCR); target 0 |
| F2 score | F-beta with β = 2 (recall weighted 2×, because a leak is worse than an over-clean) |

### 3.3 Utility metrics (did we keep research data?)
| Metric | Definition |
|---|---|
| PHI precision | elements we changed that really were PHI / all elements we changed |
| Over-cleaning rate | elements the answer key says *keep* that we removed or changed / all *keep* elements |
| Research-field retention | research-critical tags kept bit-identical / research-critical tags present. List: SliceThickness, PixelSpacing, SpacingBetweenSlices, ImagePositionPatient, ImageOrientationPatient, RepetitionTime, EchoTime, InversionTime, FlipAngle, MagneticFieldStrength, KVP, XRayTubeCurrent, ExposureTime, ConvolutionKernel, RescaleSlope, RescaleIntercept, WindowCenter, WindowWidth, Manufacturer, ManufacturerModelName, SequenceName, ScanningSequence, BodyPartExamined, Modality, Rows, Columns, BitsStored, PhotometricInterpretation, (PET) RadiopharmaceuticalInformationSequence dose/half-life, PatientAge, PatientSex, PatientWeight. Target 100% |
| Private-tag retention | non-PHI private tags kept / non-PHI private tags present (baseline that deletes all = 0%) |
| Pixel preservation | pixels outside redacted boxes bit-identical (max abs diff = 0); % image area redacted |

### 3.4 Structural / validity metrics
| Metric | Definition |
|---|---|
| DICOM validity rate | files with 0 *errors* from `dciodvfy` (dicom3tools) / all output files; warnings counted separately, and errors already present in the input are not blamed on us (compare input vs output error counts) |
| Readability | files readable by pydicom with `force=False` and pixel array decodable |
| Hierarchy preserved | #patients, #studies, #series and file→series mapping identical before/after (UID remap consistency) |
| Date-interval preserved | for each patient, the difference between any two dates (e.g. StudyDate − BirthDate, AcquisitionDate − StudyDate) is the same before and after shifting |
| Pseudonym consistency | the same original value → the same pseudonym everywhere; different values → different pseudonyms (0 collisions) |
| Determinism | running twice gives identical output hashes |

### 3.5 Audit / explainability metrics
| Metric | Definition |
|---|---|
| Audit coverage | changes found by diffing input vs output that have an audit entry / all changes; target 100% |
| Audit completeness | entries that include: file, tag, old-value hash, new value, action, reason code, PS3.15 profile option, GDPR concept; target 100% |
| Explainability | every fingerprint redaction points to the fingerprint entry it matched |

### 3.6 GDPR re-identification metrics (Phase 5)
Quasi-identifiers (QI) = attributes left in the output that an attacker could use:
age, sex, weight, height, shifted dates, modality, manufacturer, model, device serial, station name,
body part, institution-related private tags.
| Metric | Definition |
|---|---|
| Singling out | % of patients that are unique on the QI set; minimum k (k-anonymity); % patients with k < 5 |
| Linkability | attacker gets the original records (with real identities) and the de-identified set; links them by QI similarity. Metric = top-1 / top-5 linking accuracy vs random guess (1/N) |
| Inference | can an attacker recover a hidden attribute (e.g. true birth year, true study year, institution) from what remains? Accuracy vs majority-class baseline |
| GDPR label | pseudonymised if linkability is clearly above random or a key table exists; otherwise "anonymised under this attack model" (with stated limits) |

### 3.7 Efficiency metrics
| Metric | Definition |
|---|---|
| Throughput | files/second and studies/hour (headers only, and headers + OCR) |
| Scaling | throughput with 1, 4, 8, 16 worker processes |
| Peak RAM / GPU memory | measured per run |

---

## 4. Phases

### Phase 0: Setup and data acquisition
Goal: a clean, reproducible environment and verified data.

Tasks
1. Create the environment (conda, `environment.yml`); `requirements.txt` with pinned versions
   (pydicom, numpy, pandas, pyyaml, tqdm, matplotlib, pytest, easyocr, opencv-python-headless, nibabel).
2. Install `dicom3tools` (`dciodvfy`) for validity checks.
3. Install the TCIA / NBIA Data Retriever (command-line mode) and download using `.tcia` manifests:
   validation images first (7.63 GB), test images later (11.09 GB).
4. Get the answer keys (IBM Aspera Connect, 2.6 GB + 3.38 GB). Fallback: download on a laptop and `scp`.
5. Get the official MIDI-B validation/scoring script from GitHub; pin the commit hash.
6. Record licenses per collection (CC BY 4.0) in `docs/DATA.md`.
7. Build `dev-mini` (20 subjects, stratified by modality) with a fixed random seed.

Deliverables: `requirements.txt`, `docs/DATA.md`, `notebooks/00_setup_and_data.ipynb`,
`results/phase0/inventory.csv`.

| Metric | Pass criterion |
|---|---|
| Files downloaded vs manifest | 100% (exact count match: 23,921 validation images) |
| Checksums / file sizes | all match; 0 corrupted files |
| pydicom read success | 100% (unreadable files listed) |
| Answer key parsed | every answer-key file maps to an image file (100% join) |
| Scoring script | runs end-to-end on 1 subject without errors |
| Disk used | ≤ 15 GB for Phase 0 |

---

### Phase 1: Data exploration and baseline
Goal: understand where PHI lives, and set the number to beat.

Tasks
1. Inventory: subjects, studies, series, files per modality; SOP classes; transfer syntaxes
   (compressed vs uncompressed); image sizes.
2. Tag census: for every tag: how many files have it, VR, % empty, example values (hashed).
   Separate standard / private / sequence-nested tags.
3. Answer-key analysis: count of required actions by type, by tag, by modality.
   Which tags most often need action; which free-text tags carry PHI; how many pixel actions.
4. PHI location map: for each synthetic identifier, in which tags / channels it appears
   (this tells us how often PHI is *outside* its "home" tag → motivates the fingerprint).
5. Baseline 0 (sanity floor): copy files unchanged → shows what the scorer gives for "nothing done".
   Baseline A: DICOM PS3.15 Basic Application Level Confidentiality Profile only
   (remove/blank everything listed, delete all private tags, new UIDs, no date shift).
6. Baseline B: Basic Profile + Retain Longitudinal Temporal Information (date shift) + Clean
   Descriptors (what a careful standard tool would do).
7. Score both with the official script and our own metrics.

Deliverables: `notebooks/01_exploration_and_baseline.ipynb`, `results/phase1/` (tag census CSV,
action stats, plots: actions by type, PHI location heat map, baseline score table).

| Metric | Pass criterion |
|---|---|
| Tag census coverage | 100% of tags in dev-mini + validation catalogued |
| PHI location map | % of PHI found outside its home tag is reported (expected > 0 → motivation) |
| Baseline A / B action accuracy | reported per action type and modality (expected ≈ 85–95%) |
| Baseline over-cleaning rate | reported (expected high: all private tags removed) |
| Baseline leak count | reported, top-20 leak examples listed |
| DICOM validity of baseline outputs | reported (gives a reference for later phases) |
| Runtime | dev-mini baseline < 5 min |

This phase ends with a stop: I show you the baseline numbers before Phase 2.

---

### Phase 2: Policy-driven rules engine + audit log
Goal: a correct, explainable, configurable rule layer.

Tasks
1. Written policy → YAML (`configs/policy_midib.yaml`): for each tag (or tag group / VR),
   action = `remove | blank | replace | pseudonymise | date_shift | uid_remap | keep | clean_text`,
   with the PS3.15 option it comes from (Basic Profile, Retain Longitudinal Temporal Info with
   Modified Dates, Retain Patient Characteristics, Retain Device Identity, Clean Descriptors,
   Clean Structured Content, Clean Pixel Data, Retain Safe Private) and a GDPR concept label
   (direct identifier, quasi-identifier, pseudonymisation, data minimisation Art. 5(1)(c),
   anonymisation Recital 26).
2. Rules engine that walks all elements including nested sequences.
3. Pseudonymisation: keyed HMAC-SHA256 of the original value → consistent pseudonym
   (same patient → same pseudonym across files). The key is stored separately (key file = the
   reason the output is *pseudonymised* under GDPR).
4. Date shifting: one random offset per patient (derived from the key), applied to DA, DT and
   TM-linked fields; age fields kept or capped (> 89 → "090Y" per HIPAA Safe Harbor).
5. UID remapping: deterministic new UIDs under a project root; keep the study/series/instance
   hierarchy and all references (ReferencedSOPInstanceUID, FrameOfReferenceUID, etc.).
6. Audit log (one JSON-lines row per change) + per-file summary CSV.
7. Policy unit tests on hand-made tiny DICOM files.

Deliverables: `configs/policy_midib.yaml`, `src/fpdeid/rules.py`, `src/fpdeid/audit.py`,
`notebooks/02_rules_engine.ipynb`, `results/phase2/`.

| Metric | Pass criterion |
|---|---|
| Action accuracy (validation) | ≥ 97%, and > Baseline B |
| Accuracy on *keep* actions | ≥ 99.5% (little over-cleaning) |
| Research-field retention | 100% |
| DICOM validity rate | 100% (no new dciodvfy errors vs input) |
| Hierarchy preserved | 100% |
| Date-interval preserved | 100% of patients |
| Pseudonym consistency | 100%, 0 collisions |
| Audit coverage / completeness | 100% / 100% |
| Determinism | identical hashes over 2 runs |
| Error analysis | every remaining error grouped by cause (top-10 causes table) |

---

### Phase 3: PHI fingerprint for free text and private tags (novel part 1 + 2)
Goal: catch PHI that hides outside its home tag, without deleting useful data.

Tasks
1. Fingerprint builder: per patient, gather values from home tags across all their files;
   generate variants:
   - names: order swaps, initials, with/without middle name, case, `^` / `,` / space separators,
     accents removed;
   - dates: ≥ 10 formats (YYYYMMDD, DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, DD-Mon-YYYY, "March 12, 1965", …);
   - IDs: with/without leading zeros, dashes, spaces;
   - addresses / phone / email / institution: normalised forms.
2. Matcher: exact + normalised match first (fast, Aho-Corasick style), then fuzzy match
   (edit distance ≤ 1 for tokens ≥ 5 chars) for typos.
3. Generic PHI patterns for values not in the fingerprint: dates, phone, email, ID-like
   alphanumeric strings, ages, postcodes.
4. Value-aware private-tag classifier (rule based, explainable):
   remove if fingerprint/pattern hit or private creator is known to hold PHI;
   keep if numeric / code-like and no hit; otherwise follow policy.
5. Text cleaning: in free-text fields replace only the matched span (e.g.
   "Pt John Doe, follow-up" → "Pt [NAME], follow-up") when the policy allows clean-text,
   otherwise blank the field.
6. Comparison baseline: a NER-based cleaner (spaCy / Presidio) on the same fields.

Deliverables: `src/fpdeid/fingerprint.py`, `src/fpdeid/private_tags.py`,
`notebooks/03_fingerprint.ipynb`, `results/phase3/`.

| Metric | Pass criterion |
|---|---|
| Action accuracy (validation) | ≥ 99% (MIDI-B 2024 finishers: 97.91–99.93%) |
| Free-text + private-tag leak count | 0 on validation (or each remaining leak listed with cause) |
| Residual fingerprint hits in headers | 0 |
| PHI recall / precision / F2 on free-text + private fields | reported; F2 ≥ 0.98 |
| Private-tag retention (non-PHI) | ≥ 90% (baselines: 0%) |
| Over-cleaning rate | lower than Phase 2 and lower than Baseline B |
| Fingerprint vs NER | recall, precision, false positives per 1,000 fields, runtime; fingerprint expected to have higher precision at equal or higher recall |
| Variant ablation | recall with exact-only vs +variants vs +fuzzy |
| Explainability | 100% of fingerprint redactions have a matched-entry reference |

---

### Phase 4: Burned-in pixel text
Goal: remove identifying text from pixels, keep harmless text, change nothing else.

Tasks
1. Synthetic burn-in benchmark (known answer): take clean images from validation
   (all modalities), burn text with varied fonts, sizes, positions (corners, header bands),
   contrast, rotation, and overlap with anatomy. Mix of PHI text (names, IDs, dates from the
   fingerprint) and non-PHI text (L/R, "AXIAL", "5 mm", "W:400 L:40").
   Target ~2,000 images, ground-truth boxes + labels saved as JSON.
2. Which images to scan: flag candidate images (BurnedInAnnotation = YES, modalities like US / SC / DX,
   secondary captures, high-contrast corner text), OCR the rest too on a sample to measure misses.
3. Text detection + OCR: EasyOCR on GPU 0 (fallback: Tesseract on CPU), at window/level
   normalised 8-bit copies of the image; multi-frame handled frame by frame.
4. Decision: redact a box if OCR text matches the fingerprint or a PHI pattern; keep otherwise;
   option "strict" = redact all text (to compare).
5. Redaction: fill box (+ small margin) with min value in stored pixel data;
   re-encode correctly (handle compressed transfer syntaxes); set BurnedInAnnotation = NO.
6. Run on the 15 scored MIDI-B pixel actions as well.

Deliverables: `src/fpdeid/pixels.py`, `notebooks/04_pixel_text.ipynb`, `results/phase4/`
(example before/after images, metrics table).

| Metric | Pass criterion |
|---|---|
| PHI text-box recall (IoU ≥ 0.5 or character overlap) | 100% on synthetic-burn test split (each miss shown) |
| Text-box detection precision | reported; ≥ 0.9 |
| Character-level leak (OCR on output finds any PHI char string ≥ 3 chars) | 0 |
| Non-PHI text preserved | ≥ 90% in value-aware mode (strict mode = 0%, for comparison) |
| % image area redacted | reported; value-aware < strict |
| Pixels outside boxes changed | 0 (max abs diff = 0) |
| MIDI-B pixel actions | 15/15 correct |
| DICOM validity after pixel edit | 100% |
| Throughput | images/second for OCR on GPU 0; GPU memory ≤ 8 GB |
| Screening benefit | time with candidate screening vs OCR on every image, and PHI boxes missed by screening (on the OCR-everything sample) | ≥ 5× faster with 0 PHI boxes missed |

---

### Phase 5: Closed-loop auditor + privacy/utility report (novel part 3)
Goal: prove the result, on unseen data, with numbers a reviewer can trust.

Tasks
1. Auditor: independent module (does not import the cleaner's code) that rebuilds the
   fingerprint from the *original* files and scans every *output* file (all text values +
   OCR on flagged images). Output: leak list with file, tag/box, matched value type.
2. Run once on the MIDI-B test set (322 subjects), rules frozen before this run.
3. Re-identification attack for singling out / linkability / inference (Section 3.6),
   under two settings: output only, and output + public-style QIs (age, sex, study year).
   Compare a pseudonymised output (dates shifted, key file kept) with an anonymised output
   (key destroyed, dates reduced to year or intervals, ages in 5-year bins, device serials removed).
   This shows in numbers why pseudonymised data is still personal data under GDPR.
4. Privacy vs utility plot: each method (Baseline A, Baseline B, Phase 2, Phase 3, Phase 3+4,
   strict mode) as a point: x = utility (research-field + private-tag retention, 1 − over-cleaning),
   y = privacy (PHI recall, 1 − linkability).
5. Ablation table (Section 5).
6. Error analysis: every remaining miss, grouped by cause, with a sentence on why.

Deliverables: `src/fpdeid/auditor.py`, `src/fpdeid/reid_attack.py`,
`notebooks/05_auditor_and_report.ipynb`, `results/phase5/report.md` (+ plots).

| Metric | Pass criterion |
|---|---|
| Action accuracy on test set | high 90s (target ≥ 99%), all misses explained |
| Leaks in pixels + free text on test set | 0, or each listed |
| Auditor recall on planted leaks (we deliberately leave 100 known PHI values in copies of outputs) | 100% |
| Auditor false alarms | reported per 10,000 elements |
| Singling out (% unique on QIs), min k | reported with and without age/sex generalisation |
| Linkability top-1 / top-5 | reported vs random baseline 1/N |
| Inference accuracy | reported vs majority baseline |
| Research-field retention on test | 100% |
| DICOM validity on test | 100% |

---

### Phase 6: Engineering: pipeline, tests, logging, lineage
Goal: a tool someone else can run, trust and reproduce.

Tasks
1. One command: `python -m fpdeid.run --input DIR --output DIR --policy configs/policy_midib.yaml`.
2. Parallel processing per study (multiprocessing, configurable workers), resumable
   (skip files already done), memory bounded.
3. Logging: standard `logging` to file + console; per-run summary (counts, errors, timing).
4. Data lineage: `lineage.jsonl` per run: input file SHA-256 → output file SHA-256, policy
   version hash, code git commit, run timestamp, audit entry IDs.
5. Test suite (pytest): unit tests (each action, date shift, UID remap, fingerprint variants,
   pattern regexes, pixel redaction), integration test on 3 tiny DICOM files committed to
   `tests/data/` (synthetic, made by us), regression test (dev-mini score must not drop).
6. GitHub Actions workflow running pytest on every push (tests use only tiny synthetic files).

Deliverables: `src/fpdeid/run.py`, `tests/`, `.github/workflows/tests.yml`, `results/phase6/benchmark.csv`.

| Metric | Pass criterion |
|---|---|
| Tests | 100% pass; ≥ 40 test cases |
| Code coverage | ≥ 80% of `src/fpdeid` |
| Throughput (headers only) | reported in studies/hour and files/sec with 1/4/8/16 workers |
| Throughput (headers + OCR on flagged images) | reported |
| Peak RAM | ≤ 4 GB with 8 workers |
| Determinism | identical output hashes across 2 runs |
| Lineage completeness | 100% of output files traceable to input + policy + commit |
| Resume | killing a run and restarting produces the same final output |

---

### Phase 7: (Optional) Defacing head MRI
Goal: remove the face so 3D renders cannot be recognised, without touching the brain.

Tasks
1. Download a small IXI T1 subset (~30 volumes, NIfTI) to `ayra_data/ixi/`; verify license.
2. Brain-protected defacing: compute a brain mask (HD-BET or SynthStrip), dilate it by a
   safety margin, and remove (zero) only the face region (anterior-inferior to the brain,
   found from the mask + head surface), so brain voxels are guaranteed unchanged.
3. Compare with existing tools: `pydeface`, `mri_reface`/`quickshear` (if installable).
4. Face check: render the 3D head surface (marching cubes) from 8 angles;
   run a face detector (MediaPipe / RetinaFace) on the renders before and after.
5. Utility check: run a segmenter (SynthSeg or FAST) before and after; compare.

| Metric | Pass criterion |
|---|---|
| Face detection rate on renders | before ≈ 100%, after 0% (all 8 angles) |
| Brain voxels changed | 0 (guaranteed by mask) |
| Segmentation Dice before vs after (per structure, mean) | ≥ 0.99 |
| Brain volume change | < 0.5% |
| Removed face volume | reported (vs pydeface) |
| Runtime per volume | reported |

---

## 5. Ablation study (how we defend the novelty)

Run on validation (and once on test in Phase 5). Each row adds one component.

| Configuration | Action acc. | Leaks | Over-clean rate | Private-tag retention | Pixel area redacted |
|---|---|---|---|---|---|
| Baseline A: PS3.15 Basic Profile | | | | 0% | – |
| Baseline B: + options (dates, clean descriptors) | | | | 0% | – |
| Phase 2: policy engine | | | | | – |
| + NER on free text (comparison) | | | | | – |
| + fingerprint exact | | | | | – |
| + fingerprint variants + fuzzy | | | | | – |
| + value-aware private tags | | | | | – |
| + pixel OCR, strict | | | | | |
| + pixel OCR, value-aware | | | | | |
| + closed-loop auditor fixes | | | | | |

The novelty is defended if: (1) the fingerprint rows reduce leaks compared with NER at equal or better
precision; (2) the value-aware rows reduce over-cleaning without adding leaks; (3) the auditor
finds 100% of planted leaks.

---

## 6. Final deliverables checklist
- [ ] De-identified DICOM output (valid DICOM) for validation and test sets
- [ ] Per-file audit log (JSON lines) mapped to PS3.15 options and GDPR concepts
- [ ] Privacy vs utility report (`results/phase5/report.md`) with all metrics above
- [ ] Ablation table and error analysis
- [ ] Throughput benchmark and test suite
- [ ] (Optional) Defaced IXI volumes + face/Dice results
- [ ] README with how-to-run, results summary, limitations

## 7. Risks and fallbacks
| Risk | Fallback |
|---|---|
| Aspera not installable on server | download answer keys on a laptop and `scp`; or the scoring tool's own key download |
| Answer key changed since 2024 (NCI notes updates) | report which key version we used; compare with published scores carefully |
| `/root` disk full | everything (environments, caches, data) on the large disk; `PIP_CACHE_DIR` and `EASYOCR_MODULE_PATH` set to `/home` |
| OCR too slow | OCR only flagged images; batch on GPU; Tesseract fallback on CPU |
| GPU 1 busy | pin `CUDA_VISIBLE_DEVICES=0` |
| Overfitting to validation | test set run once, rules frozen before |
