# Phase 0: Setup and data

**Notebook:** [`notebooks/00_setup_and_data.ipynb`](../notebooks/00_setup_and_data.ipynb)
**Results:** [`results/phase0/`](../results/phase0/)

## What this phase is about

Before cleaning any scans, I need three things:

1. **The scans**, hospital image files in the DICOM format, with fake (synthetic) patient details written into them.
2. **The answer key**, a list made by experts that says, for every file, what *should* be removed, changed or kept.
3. **The marking tool**, the official MIDI-B scoring script, which compares my cleaned files with the answer key and gives a score.

Phase 0 gets all three, checks that nothing is missing or broken, and prepares a small subset for quick experiments.

## The data

I use the **MIDI-B Synthetic Validation** set from The Cancer Imaging Archive (TCIA), licence CC BY 4.0.
The patient details in it are fake, but they are placed exactly where real hospitals put them.

| Modality | Patients | Studies | Series | Images | Size (MB) |
|---|---|---|---|---|---|
| PET (PT) | 29 | 37 | 48 | 15,724 | 646 |
| CT | 39 | 48 | 49 | 4,406 | 2,329 |
| MRI (MR) | 52 | 55 | 60 | 3,511 | 1,122 |
| Ultrasound (US) | 24 | 25 | 25 | 167 | 1,039 |
| Digital X-ray (DX) | 22 | 25 | 27 | 34 | 453 |
| Mammography (MG) | 25 | 25 | 25 | 31 | 1,646 |
| Computed radiography (CR) | 22 | 23 | 24 | 26 | 392 |
| Structured reports (SR) | 21 | 22 | 22 | 22 | 2 |
| **Total** | **216** | **241** | **280** | **23,921** | **7,629** |

The answer key is a 2.8 GB SQLite database with **6,173,204 required checks** (one row per file, each with a list of checks).

## Checks and results

| Check (from the plan) | Target | Result |
|---|---|---|
| Files downloaded vs. what TCIA lists | 100% | **23,921 / 23,921 (100%)** |
| Files that open without error | 100% | **100%** |
| Answer-key rows that match a file | 100% | **23,921 / 23,921 (100%)** |
| Patients / UIDs present in the ID mapping tables | 100% | **216 / 216 patients, all 24,442 UIDs** |
| Scoring script runs | yes | **yes** |

Other facts I found:

- Every file has private (vendor-specific) tags.
- 44 files are multi-frame, and 20 files are JPEG 2000 compressed (the rest are uncompressed).
- **No file sets the "burned-in annotation" flag**, even though some X-rays have names written into the pixels.
  So later I cannot trust that flag; I have to look at the pixels themselves.

I also picked **dev-mini**: 20 patients (1,192 files) covering every modality, chosen with a fixed random seed.
I use it for fast experiments before running on the full set.

## Problems I solved

**1. The official download tool is a desktop application.**
TCIA normally asks you to use its "Data Retriever" program. On a remote server without a screen that is awkward,
so I wrote a small downloader ([`scripts/download_midi_b.py`](../scripts/download_midi_b.py)) that uses TCIA's public REST API.
It asks for the list of series, downloads each series as a zip, unpacks it and deletes the zip. If it stops, it can be re-run
and skips the series that are already complete.

**2. Each download connection was very slow (~64 KB/s).**
My internet speed was fine (~5 MB/s), so the limit was per connection on the server side. Downloading 16 series in parallel
made the whole download about 16× faster (the 7.6 GB set took about 2.5 hours instead of more than a day).

**3. The answer key can only be downloaded with IBM Aspera, and the Aspera port was blocked.**
Aspera uses network port 33001, which my institution's firewall blocks. I installed the command-line Aspera client and tried
the HTTP and WebSocket fallbacks, but the TCIA server does not offer them. In the end I downloaded the key on a different
network (mobile hotspot) and copied it to the server.

**4. The scoring script needs old versions of its packages.**
It pins `pydicom==2.2.1` and imports EasyOCR (which needs PyTorch). So the scorer runs in its own conda environment
(`midi-scorer`, from `environment-scorer.yml`, with a CPU-only PyTorch), and my own code runs in a modern one (`fpdeid`).

**5. The scorer finds my files through ID mapping tables.**
The scorer reads the new UID of each cleaned file and maps it back to the original with TCIA's mapping CSVs.
So my pipeline must give every UID and Patient ID exactly the new value in those tables (for example `MIDI_1_1_001`).
In a real hospital this table would be the pseudonymisation key kept by the data custodian, so this is a realistic setup.

**6. A DICOM validity checker for Linux.**
The scoring repository only contains `dciodvfy` (the standard DICOM validator from dicom3tools) for Mac and Windows.
I unpacked the Ubuntu package of dicom3tools into the project folder, without installing anything system-wide.

**7. Many input files are already slightly invalid.**
For example, the synthetic UIDs contain parts with leading zeros (`.070.`), which the DICOM rules do not allow.
So a fair validity test is: *does my output have any **new** error that the input did not have?*
I compare error types (not values) between input and output ([`src/fpdeid/validity.py`](../src/fpdeid/validity.py)).

## How to reproduce

```bash
# images (public API, no login)
python scripts/download_midi_b.py MIDI-B-Synthetic-Validation ayra_data/midi_b/validation/images
# answer key: download "MIDI-B-Answer-Key-Validation" from the TCIA collection page with IBM Aspera
# and put the .db file in ayra_data/midi_b/validation/answer_key/
jupyter nbconvert --to notebook --execute --inplace notebooks/00_setup_and_data.ipynb
```

More details on data sources and licences: [`docs/DATA.md`](DATA.md).
