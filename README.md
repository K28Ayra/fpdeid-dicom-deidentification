# FP-DeID: fingerprint-guided, closed-loop DICOM de-identification

*Erasmus Mundus EMIA, Project 2: Making hospital scans safe to share*

FP-DeID removes everything that can identify a patient from DICOM files (headers, free text, private tags, text burned into
the pixels, and optionally the face in head scans), keeps what researchers need, and measures both on the public MIDI-B
benchmark with the official scorer. Every change is written to an audit log mapped to DICOM PS3.15 and GDPR.

## 1. Setup

Tested on Ubuntu 22.04 with Python 3.10, an NVIDIA GPU (driver for CUDA 12.6) and conda. A GPU is only needed for the
OCR of burned-in text (Phase 4); everything else runs on the CPU.

```bash
git clone https://github.com/K28Ayra/fpdeid-dicom-deidentification.git
cd fpdeid-dicom-deidentification

# main environment: pipeline, notebooks, tests
conda env create -f environment.yml
conda activate fpdeid
python -m ipykernel install --user --name fpdeid --display-name "Python 3 (fpdeid)"

# external tools (official scorer, DICOM validator, face detectors, SynthStrip) into tools/
bash scripts/setup_tools.sh

# second environment for the official MIDI-B scorer (it pins old package versions)
conda env create -f environment-scorer.yml
export SCORER_PYTHON=$(conda run -n midi-scorer which python)

# check that everything works (69 tests, about 20 s, no data or GPU needed)
python -m pytest -q tests
```

If the disk with your home folder is small, create the environments somewhere else with
`conda env create -f environment.yml -p /big/disk/envs/fpdeid` and keep the data there with `export FPDEID_DATA=/big/disk/data`.

## 2. Data

The data is **not** in this repository: the MIDI-B data is about 19 GB plus 6 GB of answer keys, far above what GitHub
(also with Git LFS) can store. Everything goes into `ayra_data/` (or `$FPDEID_DATA`). Git LFS is not used: without the
data, the largest file in the repository is a 1.6 MB figure.

```bash
# MIDI-B images (public TCIA API, no login): validation 7.6 GB, test 11.1 GB
python scripts/download_midi_b.py MIDI-B-Synthetic-Validation ayra_data/midi_b/validation/images
python scripts/download_midi_b.py MIDI-B-Synthetic-Test       ayra_data/midi_b/test/images

# IXI head MRI for the defacing phase (30 scans, 230 MB)
bash scripts/download_ixi.sh
```

Two parts must be downloaded by hand from the [MIDI-B collection page](https://www.cancerimagingarchive.net/collection/MIDI-B-Test-MIDI-B-Validation/):

| File | Put it in |
|---|---|
| `MIDI-B-Patient-Mapping-Validation.csv`, `MIDI-B-UID-Mapping-Validation.csv` (and the `-Test` versions) | `ayra_data/midi_b/` |
| Answer key `MIDI-B-Answer-Key-Validation.db` (IBM Aspera download, needs port 33001) | `ayra_data/midi_b/validation/answer_key/` |
| Answer key `MIDI-B-Answer-Key-Test.db` | `ayra_data/midi_b/test/answer_key/` |

Details and licences (MIDI-B: CC BY 4.0, IXI: CC BY-SA 3.0): [docs/DATA.md](docs/DATA.md).

## 3. How to run everything

```bash
conda activate fpdeid
export SCORER_PYTHON=$(conda run -n midi-scorer which python)
export CUDA_VISIBLE_DEVICES=0                       # the GPU used for OCR

# the notebooks, in order (each one saves its tables and figures in results/phaseN/)
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 00_setup_and_data.ipynb             # file checks, dev-mini (~10 min)
jupyter nbconvert --to notebook --execute --inplace 01_exploration_and_baseline.ipynb   # census + baselines (~1 h)
jupyter nbconvert --to notebook --execute --inplace 02_rules_engine.ipynb               # ~30 min
jupyter nbconvert --to notebook --execute --inplace 03_fingerprint_text_private.ipynb   # ablation, 8 scorer runs (~3 h)
jupyter nbconvert --to notebook --execute --inplace 04_burned_in_text.ipynb             # OCR on GPU (~1 h)
jupyter nbconvert --to notebook --execute --inplace 05_auditor_and_report.ipynb         # test set + GDPR tests (~2 h)
jupyter nbconvert --to notebook --execute --inplace 06_engineering.ipynb                # benchmarks (~45 min)
jupyter nbconvert --to notebook --execute --inplace 07_defacing.ipynb                   # defacing (~6 h, CPU renders)
cd ..
```

Notebook 05 runs the full pipeline and the official scorer on both MIDI-B splits through `scripts/run_split.py`;
they can also be run alone:

```bash
python scripts/run_split.py validation       # pipeline + auditor + official scorer (~35 min)
python scripts/run_split.py test             # the held-out test set (~45 min)
```

**De-identify your own DICOM files** with one command:

```bash
export FPDEID_SECRET="a long secret kept by the hospital"     # the pseudonymisation key
python -m fpdeid.run --input /path/to/dicoms --output /path/to/output
#   --no-pixels   skip the OCR of burned-in text (no GPU needed)
#   --workers 16  number of CPU processes
#   --uid-map uids.csv --patient-map patients.csv   use existing pseudonym tables (id_old,id_new)
```

The output folder contains the de-identified files, the audit logs (`audit*.jsonl`), the lineage of every file
(`lineage.jsonl`), `run.log` and `run_summary.json`. A stopped run continues where it stopped when started again.

The notebooks are also stored as plain Python files in `notebooks/src_py/`. After editing one, rebuild the notebook with
`python -m jupytext --to ipynb --update -o notebooks/NAME.ipynb notebooks/src_py/NAME.py`.

## 4. The idea

Standard tools clean a DICOM file one field at a time with fixed rules ("delete Patient Name, keep Series Description").
But in MIDI-B, **98.8% of files contain the patient's own name or ID inside some other field**, for example a Series
Description like *"PET-CT SERIES for Peggy Peters"*. A fixed rule can only keep that field (a leak) or delete it (losing
useful information). FP-DeID first builds a **fingerprint** of each patient from the fields where identity is known to
live (name, ID, birth date, ...), then searches for it everywhere else (free text, private tags, sequences and text read
from the pixels) and removes only the matching words. An independent **auditor** re-checks every output file, and GDPR
re-identification tests measure what is left.

## 5. Results

All rows except the last are on the MIDI-B **validation** set (216 patients, 23,921 files, 6,173,204 required checks); the
last row is the held-out **test** set, cleaned once with the rules frozen. Official MIDI-B scorer (70% HIPAA, 20% DICOM,
10% TCIA best practice). Most failed checks are vendor private values that TCIA removes but that hold no PHI; the real
leaks are listed in each phase document.

| Method | Weighted score | Privacy checks failed | Utility checks failed | Valid DICOM |
|---|---|---|---|---|
| B0: replace IDs only | 61.51% | 454,324 | 55,371 | 100% |
| BA: DICOM PS3.15 Basic Profile | 95.31% | 21,358 | 1,670,457 | 99.8% |
| BB: Basic Profile + standard options | 95.87% | 31,131 | 1,458,139 | 99.5% |
| Phase 2: policy rules engine | 99.15% | 4,577 | 302,210 | 100% |
| Comparison: rules + NER (spaCy) for free text | 98.80% | 11,119 | 163,270 | - |
| Phase 3: rules + PHI fingerprint | 99.828% | 5,198 | 55,003 | 100% |
| Phase 4: + burned-in text in pixels | 99.831% | 5,177 | 55,004 | 100% |
| **Phase 5: + closed-loop independent auditor** | **99.831%** | **5,177** | **55,004** | **100%** |
| **Held-out test set (322 patients, 29,660 files), run once** | **99.805%** | 13,765 | 65,778 | **100%** |

For reference, the ten teams that finished the MIDI-B challenge (MICCAI 2024) scored 97.91-99.93% on the test set.

## 6. Phases

| Phase | What it does | Doc |
|---|---|---|
| 0. Setup and data | Download the MIDI-B data, answer key and scorer; check that everything is complete | [phase0_setup_and_data.md](docs/phase0_setup_and_data.md) |
| 1. Exploration and baselines | Find where PHI hides; score three standard baselines | [phase1_exploration_and_baselines.md](docs/phase1_exploration_and_baselines.md) |
| 2. Policy rules engine + audit log | Written policy to rules; consistent pseudonyms and date shifts; audit log mapped to PS3.15 and GDPR | [phase2_rules_engine.md](docs/phase2_rules_engine.md) |
| 3. PHI fingerprint | Find each patient's identifiers in free text and private tags; keep useful data | [phase3_fingerprint.md](docs/phase3_fingerprint.md) |
| 4. Burned-in pixel text | OCR on images; hide only text that identifies the patient | [phase4_burned_in_text.md](docs/phase4_burned_in_text.md) |
| 5. Auditor + privacy/utility report | Independent re-check; GDPR re-identification tests; final test-set score | [phase5_auditor_and_report.md](docs/phase5_auditor_and_report.md) |
| 6. Engineering | One-command pipeline, tests, logging, data lineage, speed | [phase6_engineering.md](docs/phase6_engineering.md) |
| 7. Defacing (optional) | Remove faces from head MRI without touching the brain | [phase7_defacing.md](docs/phase7_defacing.md) |

The plan with every metric and pass criterion is in [docs/PLAN.md](docs/PLAN.md).

### Phase 0: Setup and data

I downloaded all 23,921 validation files (7.6 GB) through TCIA's public API with a parallel downloader, plus the
2.8 GB answer key and the official scorer. 100% of the files are present and readable, and every answer-key row matches a
file. The scorer runs in its own environment, and the Linux DICOM validator (`dciodvfy`) is unpacked locally. A 20-patient
*dev-mini* subset is used for fast experiments. No file flags burned-in text, so the pixels have to be checked directly.

### Phase 1: Exploration and baselines

All 1,444 tag paths (621 private) are catalogued. The patient's name or ID appears outside its home field in 98.8% of
files, and *Series Number* often holds a date. The official PS3.15 rule table (parsed from the NEMA standard) is
implemented as three baselines. The best one scores **95.87%**: it removes almost all identifiers (99.99% of HIPAA checks)
but deletes about a third of the useful data (private PET parameters, study and series descriptions).

### Phase 2: Policy rules engine and audit log

The de-identification policy is a readable YAML file, and a rules engine applies it to every field, including fields
inside sequences and vendor private tags. Private tags are decided by **value type** (numbers and codes kept, text
cleaned), **every UID in the pseudonym key is replaced wherever it appears**, and a **value check** catches dates hidden in
number fields (Series Number, Unix time stamps). Every change goes to an audit log with its reason, PS3.15 option and
GDPR concept, without storing any original value. Score: **99.15%**, with 100% valid DICOM, 100% research fields kept,
100% date intervals and UID consistency, 100% audit coverage and byte-identical repeat runs, at about 1,350 files per second.

### Phase 3: The PHI fingerprint

Each patient's identity is expanded into all the ways it can be written (name orders, 19 date formats, IDs with and
without dashes, hospital acronyms such as *"CHAJH"* for *"Compton, Holmes and Jones Hospital"*), plus a few checked backup
patterns (SSN, phone, *"Dr. X"*, staff initials, postal addresses, site codes). Descriptions and private text are
**cleaned word by word** instead of deleted, and a data-driven guard protects ordinary words. Compared with a standard NER
cleaner using the same rules and patterns, FP-DeID leaks **53% less** and removes **66% less useful data**.
Score: **99.83%**, with 0 residual fingerprint hits, 100% of removals explained and 100% valid DICOM.

### Phase 4: Burned-in text in the pixels

OCR (EasyOCR on GPU) reads each image in three views (normal, bright text, dark text), joins the pieces into lines and
checks every line with the same fingerprint (fuzzy, to allow OCR mistakes), label words and dates. Technologist initials
on X-rays are hidden; radiology markers (L/R, PORTABLE, scale, window) are kept, and only the boxes of identifying lines
change. On a synthetic benchmark of 400 real images with 1,647 known texts, only **2 of 540** PHI words stay readable,
while **6x more useful text** is kept than with blacking out all text. On MIDI-B, **27 of 35** burned-in checks pass (from 6),
and series screening keeps the OCR to 3% of the files.

### Phase 5: Independent auditor, GDPR tests and the test set

The **auditor** shares no code with the cleaner: it rebuilds each patient's identity from the original files and searches
every output value (sequences, private tags, readable bytes, and its own OCR of the pixels). It finds **all 120**
identifiers planted back into copies of clean files, and in the pipeline it removes every hit (0 on validation, 55 on test).
**GDPR re-identification tests** show that the normal output is *pseudonymised*, not anonymous: an insider holding the
hospital records links **98%** of patients (random: 0.3%) through age, weight, scanner and scan gaps, while the date shift
stops the true scan year from being guessed. A k-anonymised release (k = 5) cuts linking to 7% but keeps only about a
quarter of the quasi-identifier information. On the **held-out test set**, run once, FP-DeID scores **99.805%** with 100%
valid DICOM, research fields and audit coverage. The 921 remaining text leaks (0.1% of privacy checks) are listed by cause.

### Phase 6: Engineering

One command (`python -m fpdeid.run`) runs fingerprints, headers, pixels, the auditor and the lineage, with logging and a run
summary. A stopped run resumes to byte-identical files, and `lineage.jsonl` links every output file to its input, the
policy and the code version by SHA-256. 69 tests (83% coverage) run on GitHub Actions on every push. The header part cleans
**610 files/s (about 22,000 studies per hour)** with 16 workers and uses 1.15 GB of memory with 8 workers; the full tool
with OCR takes 30 minutes for the validation set, and its output is byte-identical to the scored Phase 5 output.

### Phase 7: Brain-protected defacing (optional)

SynthStrip finds the brain, a 10 mm margin protects it, and only the region in front of the brain between chin and eyebrow
level is removed, so **0 brain voxels are changed by construction**. The same code defaces a DICOM head series with
identical results and unchanged headers. On 20 IXI test scans rendered from 8 directions and checked by two face
detectors, 7 of 160 views are still flagged (quickshear: 36), and none of them shows a face when checked by eye. A
skull-stripper run on the defaced image finds a brain 0.7% different (quickshear 0.42%), so brain masks should be made
before defacing.

## 7. Repository layout

```
├── README.md
├── environment.yml, environment-scorer.yml, requirements.txt, pyproject.toml
├── docs/            plan, data sources, one document per phase
├── notebooks/       one notebook per phase (with outputs); src_py/ has the same notebooks as .py files
├── src/fpdeid/      the package used by the notebooks and the command line tool
├── configs/         rule tables and policies
├── scripts/         downloads, tool setup, run_split.py (pipeline + scorer for one MIDI-B split)
├── tests/           unit and integration tests (tests/data: 3 tiny synthetic DICOM files)
├── results/         result tables and figures, one folder per phase
├── tools/           external tools (made by scripts/setup_tools.sh, not in git)
└── ayra_data/       data and large outputs (not in git)
```
