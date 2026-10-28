# Phase 6: Engineering: one command, tests, logging, lineage, speed

**Notebook:** [`notebooks/06_engineering.ipynb`](../notebooks/06_engineering.ipynb)
**Code:** [`src/fpdeid/run.py`](../src/fpdeid/run.py) (the one-command tool),
[`scripts/make_test_data.py`](../scripts/make_test_data.py) (synthetic test files)
**Tests:** [`tests/`](../tests/) (62 tests), [`.github/workflows/tests.yml`](../.github/workflows/tests.yml) (GitHub Actions)
**Results:** [`results/phase6/`](../results/phase6/)

## What this phase is about

Phases 0–5 proved that the method works. Phase 6 turns it into a **tool someone else can run, trust and reproduce**:
one command, clear logs, a record of where every output file came from, a test suite that runs on every push, and
measured speed and memory.

## The one command

```bash
python -m fpdeid.run --input  /path/to/original/dicoms \
                     --output /path/to/deidentified \
                     [--policy configs/policy.yaml] \
                     [--uid-map uids.csv --patient-map patients.csv] \
                     [--workers 16] [--no-pixels] [--no-auditor]
```

It finds every DICOM file below `--input` (by the `DICM` marker, whatever the file name) and runs five steps:

| Step | What it does |
|---|---|
| 1. fingerprints | reads every header once and builds each patient's PHI fingerprint (Phase 3) |
| 2. headers | cleans every file with the rules + fingerprint (Phases 2–3), in parallel worker processes |
| 3. pixels | OCR on images that may have burned-in text; hides only identifying lines (Phase 4, GPU) |
| 4. auditor | the independent auditor checks every output header and removes what it finds (Phase 5) |
| 5. lineage | `lineage.jsonl`: one line per file linking output to input, policy, code version and audit entries |

The pseudonym tables are optional. Without them, every pseudonym comes from a secret (the environment variable
`FPDEID_SECRET`): the same input always gives the same pseudonym, and only the holder of the secret can repeat it.
The secret and the tables are the **pseudonymisation key** of GDPR Art. 4(5); they stay with the hospital and are never
written into the output folder. (The MIDI-B results in this project use the TCIA tables and the default secret, so they
can be repeated exactly; for real data a new, private secret must be set.)

What the output folder contains:

| File | Content |
|---|---|
| `<patient>/<series>/<instance>.dcm` | the de-identified files (all names are pseudonyms) |
| `audit.jsonl` | every change to every header: tag, action, reason, PS3.15 option, GDPR concept (no original values) |
| `audit_pixels.jsonl` | every image that was read by OCR: each text line found, and why it was hidden or kept |
| `audit_auditor_fixes.jsonl` | every value the independent auditor removed |
| `lineage.jsonl` | input SHA-256 → output SHA-256, policy SHA-256, code version, run time, audit line numbers |
| `run.log`, `run_summary.json` | the log of the run (also printed on the console) and its counts, times and peak memory |
| `ocr_cache.jsonl` | OCR results, kept so that a stopped run does not read the same image twice |

## Resume, determinism, lineage

- **Resume:** each cleaned file is written to `audit.jsonl` as soon as it is done. If the run is stopped (crash, power cut,
  `kill`), the same command skips the finished files, cleans the rest, reuses the saved OCR results, and repeats the
  auditor and lineage steps for all files. A half-written last line of the log is ignored.
- **Determinism:** pseudonyms and date shifts come from the key, never from chance or from the order of the files,
  so the result does not depend on how many workers are used or on which file is cleaned first.
- **Lineage:** every output file can be traced back to the exact input file, policy file and code version that made it,
  and to its entries in the audit logs.

## Tests

62 tests, run with `python -m pytest -q tests`, and on GitHub Actions on every push
(`.github/workflows/tests.yml`; they need no data download and no GPU):

| Test file | What it checks |
|---|---|
| `test_baselines.py`, `test_engine.py` | every action (remove, empty, dummy, pseudonymise, shift, keep), UID remapping, sequences, private tags by value type, dates hidden in numbers |
| `test_fingerprint.py` | name orders, date and ID variants, acronyms, patterns, the common-word guard |
| `test_pixels.py` | which OCR lines are hidden or kept, joining pieces into lines, only the box changes |
| `test_auditor.py` | the auditor's identity, matching in other forms, no match inside longer words, k-anonymity and linking |
| `test_evaluation_tools.py` | scorer helpers, independent checks, planted leaks + closed loop, re-identification tables, the pixel step with a given OCR result, the benchmark tools, the split runner |
| `test_run.py` | **integration**: the whole one-command run on 3 tiny synthetic DICOM files (`tests/data/input`, made by `scripts/make_test_data.py`) |

The integration tests check, on files that hide the patient's identity in a description, a comment, a private tag and
a date-like Series Number: that **none** of 13 identifiers is left anywhere in the output bytes, that research fields stay
the same, that two runs give byte-identical files, that a stopped run resumes to the same result, that the lineage hashes
are correct, and that one patient keeps one pseudonym and one date shift.

Code coverage is **83%** of `src/fpdeid` (lowest: the `dciodvfy` wrapper, which needs the external validator installed).

## Measurements on the validation set (23,921 files, 216 patients, 241 studies)

### Speed and memory, headers only

| Worker processes | Header step | Whole run (all steps) | Studies per hour | Peak memory |
|---|---|---|---|---|
| 1 | 84 files/s | 56 files/s | 2,040 | 0.70 GB |
| 4 | 358 files/s | 215 files/s | 7,802 | 0.98 GB |
| 8 | 695 files/s | 377 files/s | 13,685 | **1.15 GB** |
| 16 | 1,293 files/s | 610 files/s | 22,133 | 1.68 GB |

![Benchmark](../results/phase6/benchmark.png)

The header step scales almost linearly with the number of workers (15× faster with 16 workers). The whole run scales
less, because finding the files and building the fingerprints are partly sequential.

### The full tool (headers + OCR on GPU 0 + auditor, 16 workers)

| Step | Seconds |
|---|---|
| fingerprints | 16 |
| headers | 19 |
| pixels (OCR on 710 images, 3% of the files) | 1,683 |
| auditor | 65 |
| lineage | 4 |
| **total** | **1,787 s ≈ 30 min (13.4 files/s, 485 studies/hour)**, peak memory 3.8 GB |

OCR is 94% of the time. The header part alone could clean about 22,000 studies per hour.

## Pass criteria

| Check | Target | Result |
|---|---|---|
| Tests | 100% pass, ≥ 40 tests | **62 passed**, 0 failed |
| Code coverage of `src/fpdeid` | ≥ 80% | **83%** |
| Throughput, headers only | reported for 1/4/8/16 workers | 56 → 610 files/s; 22,133 studies/hour |
| Throughput, headers + OCR + auditor | reported | 13.4 files/s; 485 studies/hour |
| Peak memory with 8 workers | ≤ 4 GB | **1.15 GB** |
| Identical output, 8 vs 16 workers | 100% | **100%** (23,921 / 23,921 files) |
| Lineage correct and complete | 100% | **100%** (both SHA-256 checked for every file) |
| Resume after a kill | same final output | **100% identical**, 0 extra files (killed after 1,860 files) |
| One-command output = the output scored in Phase 5 | 100% | **100%** (23,921 / 23,921 byte-identical, so it scores 99.831%) |

## Problems I solved

**1. The memory grew with the number of workers (11.4 GB with 8 workers).**
One worker alone needed only 0.2 GB, so the workers were not the problem. The main process read the whole audit log
(394 MB of text, about 2.5 GB as Python objects) to find finished files, and then started new worker processes for the
later steps. A new worker starts as a copy of the main process, so every worker carried a copy of the audit log. Now the
log is read **one line at a time** and only the four values that are needed (line, input, output, error) are kept.
With 8 workers the peak is now 1.15 GB, and the full run with the OCR model 3.8 GB (before: 37 GB as measured).

**2. Measuring memory honestly.**
Adding up the memory of each process counts memory that they share (the program code, libraries) once per process.
I measure **PSS** (proportional set size), which splits shared memory between the processes that use it, so the total
is the real use of the machine.

**3. A faster start method was slower.**
Starting workers as fresh processes (`forkserver`) instead of copies (`fork`) should have saved memory, but it made the
header step twice as slow (74 s instead of 35 s). Once the main process was small (problem 1), copies were small too,
so I kept the faster method.

**4. A run must survive being stopped.**
Each file's audit line is written and flushed as soon as the file is done, and the OCR results are saved line by line.
The test kills the whole run (all worker processes, like a power cut) in the middle and starts it again; the final
23,921 files are byte-identical to a run that was never stopped.

**5. Tests that run anywhere.**
The real data cannot be put on GitHub, so I made 3 tiny synthetic DICOM files (16 × 16 pixels, invented identities hidden
in the same kinds of places as in MIDI-B). The pixel step is tested with a given OCR result instead of the OCR model,
so the tests need no GPU and finish in about 15 seconds.

**6. Drawing text into 16-bit images.**
The drawing function of OpenCV only works on 8-bit images, so the planted-leak test could not write into 16-bit X-rays.
The text is now drawn on an 8-bit mask and then copied into the real pixels with the image's brightest value (the same
method as the Phase 4 benchmark).

## How to reproduce

```bash
python -m pytest -q --cov=fpdeid tests                                                   # ~15 s
CUDA_VISIBLE_DEVICES=0 jupyter nbconvert --to notebook --execute --inplace notebooks/06_engineering.ipynb   # ~45 min
```
