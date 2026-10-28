# Data

No data is stored in this repository. Everything lives in `data/` (git-ignored).

## MIDI-B (main dataset)

- Collection page: https://www.cancerimagingarchive.net/collection/MIDI-B-Test-MIDI-B-Validation/
- DOI: https://doi.org/10.7937/cf2p-aw56
- Licence: CC BY 4.0 (all MIDI-B collections). Synthetic patient identifiers, no real PHI.
- Data usage: TCIA Data Usage Policy.

| Part | Size | How we download it |
|---|---|---|
| MIDI-B-Synthetic-Validation images (216 subjects, 280 series, 23,921 images) | 7.63 GB | `python scripts/download_midi_b.py MIDI-B-Synthetic-Validation data/midi_b/validation/images` (public TCIA REST API, no login) |
| MIDI-B-Synthetic-Test images (322 subjects, 29,660 images) | 11.09 GB | same script with `MIDI-B-Synthetic-Test` |
| Validation answer key (SQLite `.db` in a `.zip`) | 169 MB zip | IBM Aspera (Faspex package 1043), see note below |
| Test answer key | 221 MB zip | IBM Aspera (Faspex package 1042) |
| Patient / UID mapping CSVs | 3 MB | direct links on the collection page (`MIDI-B-Patient-Mapping-*.csv`, `MIDI-B-UID-Mapping-*.csv`) |
| Scoring script | - | https://github.com/CBIIT/MIDI_validation_script (cloned to `tools/` by `scripts/setup_tools.sh`) |

Answer key note: Aspera needs TCP/UDP port 33001. On our server this port is blocked, so the answer keys
were downloaded in a browser (with IBM Aspera Connect) on a laptop and copied to
`data/midi_b/validation/answer_key/` and `data/midi_b/test/answer_key/`.
The page labels are easy to mix up: check the package title says *Validation* or *Test*.

## Folder layout

```
data/
├── midi_b/
│   ├── MIDI-B-Patient-Mapping-Validation.csv
│   ├── MIDI-B-UID-Mapping-Validation.csv
│   ├── validation/
│   │   ├── series_list.json          (from the TCIA API)
│   │   ├── images/<PatientID>/<SeriesUID>/*.dcm
│   │   └── answer_key/*.db
│   └── test/ ...
├── derived/      (tables we make: file table, tag census with examples, scorer results)
└── outputs/      (de-identified files; deleted after scoring to save disk)
```

## Optional: IXI (Phase 7, defacing)

- https://brain-development.org/ixi-dataset/, licence CC BY-SA 3.0.
- 30 T1 scans (230 MB): `bash scripts/download_ixi.sh` puts them in `data/ixi/t1/`.

## Citation

*Medical Image De-Identification Benchmark (MIDI-B)*, The Cancer Imaging Archive,
https://doi.org/10.7937/cf2p-aw56 (use the exact citation text given on the collection page).
