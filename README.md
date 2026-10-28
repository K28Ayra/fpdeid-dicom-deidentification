# FP-DeID: fingerprint-guided, closed-loop DICOM de-identification



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

Standard tools clean a DICOM file field by field with fixed rules. In MIDI-B, 98.8% of the files also have the patient's
name or ID in some other field, for example *"PET-CT SERIES for Peggy Peters"* in Series Description, and a fixed rule can
only keep that field or delete it. FP-DeID reads each patient's identity from the header first, writes it out in the forms
it takes in text (name orders, date formats, ID variants, hospital acronyms) and removes only those words, in free text,
private tags, sequences and OCR'd pixel text. A separate auditor, which shares no code with the cleaner, checks the output.

## 5. Results

Validation set: 216 patients, 23,921 files, 6,173,204 checks. Test set: 322 patients, 29,660 files, run once with the
rules frozen. Scores from the official MIDI-B scorer (70% HIPAA, 20% DICOM, 10% TCIA best practice).

| Method | Weighted score | Privacy checks failed | Utility checks failed | Valid DICOM |
|---|---|---|---|---|
| B0: replace IDs only | 61.51% | 454,324 | 55,371 | 100% |
| BA: PS3.15 Basic Profile | 95.31% | 21,358 | 1,670,457 | 99.8% |
| BB: Basic Profile + options | 95.87% | 31,131 | 1,458,139 | 99.5% |
| Rules engine (phase 2) | 99.15% | 4,577 | 302,210 | 100% |
| Rules + spaCy NER for text | 98.80% | 11,119 | 163,270 | - |
| Rules + fingerprint (phase 3) | 99.828% | 5,198 | 55,003 | 100% |
| + pixel text (phase 4) | 99.831% | 5,177 | 55,004 | 100% |
| + auditor (phase 5) | 99.831% | 5,177 | 55,004 | 100% |
| Test set, final pipeline | 99.805% | 13,765 | 65,778 | 100% |

Most of the failed privacy checks are private values that TCIA removes but that hold no PHI. On the test set, 921 real
leaks remain (mostly hospital names that are not in the patient's own header, and 10-digit phone numbers); they are listed
in [phase 5](docs/phase5_auditor_and_report.md). The ten finishing teams of the MIDI-B challenge (MICCAI 2024) scored
97.91-99.93% on the same test set.

The normal output is pseudonymised, not anonymous: an attacker with the hospital's original records links 98.4% of the
test patients from age, weight, scanner and scan gaps alone.

## 6. Phases

| | Doc | Notebook |
|---|---|---|
| 0. Data, scorer, validator | [phase0](docs/phase0_setup_and_data.md) | `00_setup_and_data` |
| 1. Where PHI is; three baselines | [phase1](docs/phase1_exploration_and_baselines.md) | `01_exploration_and_baseline` |
| 2. Policy rules engine and audit log | [phase2](docs/phase2_rules_engine.md) | `02_rules_engine` |
| 3. PHI fingerprint for text and private tags | [phase3](docs/phase3_fingerprint.md) | `03_fingerprint_text_private` |
| 4. Burned-in text in pixels | [phase4](docs/phase4_burned_in_text.md) | `04_burned_in_text` |
| 5. Auditor, GDPR tests, test set | [phase5](docs/phase5_auditor_and_report.md) | `05_auditor_and_report` |
| 6. One command, tests, lineage, speed | [phase6](docs/phase6_engineering.md) | `06_engineering` |
| 7. Defacing head MRI (optional) | [phase7](docs/phase7_defacing.md) | `07_defacing` |

The original plan with all metrics and targets is in [docs/PLAN.md](docs/PLAN.md). Some targets were missed; each phase
doc says which and why.

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
