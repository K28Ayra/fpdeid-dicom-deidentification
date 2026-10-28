# Phase 0: Setup and data

Notebook: [`00_setup_and_data.ipynb`](../notebooks/00_setup_and_data.ipynb) · results in [`results/phase0/`](../results/phase0/)

I work with the MIDI-B Synthetic Validation set from TCIA (CC BY 4.0). The patient details are fake, but they sit in the
places where real hospitals put them. Getting the images was easy; getting the answer key took most of the time.

| Modality | Patients | Series | Images | Size (MB) |
|---|---|---|---|---|
| PET | 29 | 48 | 15,724 | 646 |
| CT | 39 | 49 | 4,406 | 2,329 |
| MR | 52 | 60 | 3,511 | 1,122 |
| US | 24 | 25 | 167 | 1,039 |
| DX | 22 | 27 | 34 | 453 |
| MG | 25 | 25 | 31 | 1,646 |
| CR | 22 | 24 | 26 | 392 |
| SR | 21 | 22 | 22 | 2 |
| Total | 216 | 280 | 23,921 | 7,629 |

All 23,921 files arrived and open without error, every answer-key row (6,173,204 checks in a 2.8 GB SQLite file) matches a
file, and all 216 patients and 24,442 UIDs are in TCIA's mapping tables. 44 files are multi-frame and 20 are JPEG 2000.
None of them sets *Burned In Annotation*, although some X-rays clearly have names in the pixels, so that flag is useless
here and Phase 4 has to look at the images.

For quick experiments I took 20 patients (1,192 files, every modality, fixed seed) and call them dev-mini.

## Downloading

TCIA's Data Retriever is a desktop program, and I work on a remote server, so I wrote
[`download_midi_b.py`](../scripts/download_midi_b.py) against the public NBIA REST API. One connection gave about
64 KB/s even though the line does ~5 MB/s; with 16 series in parallel the 7.6 GB came down in about 2.5 hours.
Later, one very large test-set series kept getting cut off as a zip, so
[`download_series_by_image.py`](../scripts/download_series_by_image.py) fetches it image by image. The first copy of the
UID mapping CSV was also silently truncated (my `curl --max-time 120` stopped it), so I downloaded it again in full.

The answer keys are only offered through IBM Aspera, which needs port 33001. Our firewall blocks it. I tried the Aspera
command-line client and its HTTP and WebSocket fallbacks; TCIA does not serve those. In the end the keys were downloaded
on a mobile hotspot and uploaded to the server, which took several hours.

The system disk of the machine was nearly full, so environments, caches and data are kept on the larger data disk.

## The scorer and the validator

The official CBIIT scorer pins `pydicom==2.2.1` and imports EasyOCR, so it gets its own conda environment
(`environment-scorer.yml`). It finds my output files through TCIA's mapping tables, which means my pipeline has to use
exactly those pseudonyms (for example `MIDI_1_1_001`). That is also what a hospital's pseudonymisation key would be.

The scoring repository only ships `dciodvfy` for Mac and Windows; I unpacked the Ubuntu dicom3tools package into
`tools/`. Many input files already fail it (the synthetic UIDs contain parts like `.070.`), so in
[`validity.py`](../src/fpdeid/validity.py) a file only counts as broken if my output has an error type the input did not have.
