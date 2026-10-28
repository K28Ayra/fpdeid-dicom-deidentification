# Phase 4: Burned-in text in the pixels

**Notebook:** [`notebooks/04_burned_in_text.ipynb`](../notebooks/04_burned_in_text.ipynb)
**Code:** [`src/fpdeid/pixels.py`](../src/fpdeid/pixels.py), [`src/fpdeid/burnin.py`](../src/fpdeid/burnin.py) (synthetic benchmark),
[`src/fpdeid/pixel_eval.py`](../src/fpdeid/pixel_eval.py) (metrics)
**Tests:** [`tests/test_pixels.py`](../tests/test_pixels.py)
**Results:** [`results/phase4/`](../results/phase4/)

## What this phase is about

Some images have the patient's details **written into the picture itself**, for example an X-ray with
*"JONES ERIN [F] 01.12.2019 / DOB: 11.12.1937"* in the corner, or the technologist's initials. Cleaning the header does not
remove this; the pixels must be changed.

The simple solutions are bad for research: blacking out **all** text also removes useful labels (*L/R* side markers,
*PORTABLE*, *SUPINE*, scale bars), and blacking out fixed **bands** at the top and bottom removes image content and still misses
text in other places.

My approach reuses the idea of Phase 3: **read the text, and hide only what identifies a person.**

## How it works

1. **Read the text (OCR)** with EasyOCR on the GPU. Burned-in text is usually written with the brightest (or darkest) value,
   so the image is read in **three views**: a normal view, a *bright-text* view that stretches only the top of the brightness
   range, and a *dark-text* view. Small images (e.g. 128-pixel PET slices) are enlarged first.
2. **Join the pieces into lines**, because OCR often splits *"ID: 4107422692"* into *"ID:"* and *"4107422692"*.
3. **Decide per line:**
   - matches the patient's **fingerprint** (Phase 3) → hide. OCR makes small mistakes (*"PALDWIN"* for *BALDWIN*,
     *"8787849515"* for *8787849915*), so the match may also be **fuzzy**;
   - contains a label word (*DOB, ID, MRN, NAME, PATIENT*) or a date → hide;
   - on an X-ray, a single short word read with good confidence that looks like **initials** (*JT, SWU, JKR*) → hide,
     unless the same code appears in the images of many patients (then it is a screen or device word);
   - known radiology markers (*L, R, AP, PORTABLE, SUPINE, 5 mm, W:400 L:40, MLO, ...*) and everything else → keep.
4. **Hide** each chosen line by filling its box (plus a margin of 20% of the text height) with the image's black value.
   **No pixel outside the hidden boxes changes**, and images where nothing is hidden are not rewritten at all.
5. **Screening for speed:** X-ray, mammography and ultrasound images are always read. For CT/MR/PET series only the first,
   middle and last slice are read; the rest of the series is read only if one of them has text that would be hidden.

Every image that was read gets an audit entry: every line found, its confidence, and why it was hidden or kept.

## The synthetic benchmark

The MIDI-B answer key checks only **35** burned-in text regions, too few to measure a method. So I built a benchmark where
the answer is known: **400 real images** from the validation set (CR, DX, MG, US, and slices of CT, MR and PET), with
**1,647 pieces of text** written into the pixels at the corners and edges, with different sizes and brightness:
919 PHI texts made from the patient's own identity (name line, ID, date of birth in 3 formats, hospital name or acronym)
and 728 harmless texts (markers, views, scale, window values). Patients were split 70% / 30% into a tuning set and a test set.

![Synthetic examples](../results/phase4/synthetic_examples.png)

### Results on the test split (patients never used for tuning)

| Method | PHI boxes hidden | PHI words still readable after hiding | Harmless text kept | Hidden boxes that are PHI | Image area hidden |
|---|---|---|---|---|---|
| Black out all text | 74.4% | **0** of 540 | 13.5% | 41.1% | 30.7% |
| Black out top/bottom bands | 50.4% | 35 of 540 | 4.8% | 69.1% | 29.2% |
| **FP-DeID (value-aware)** | 63.0% | **2** of 540 | **79.2%** | **96.9%** | **14.7%** |

- FP-DeID keeps **6× more useful text** than "black out all text" and hides **half the image area**, while only **2 of 540**
  PHI words remain readable afterwards.
- "PHI boxes hidden" is a strict geometric measure (≥ 90% of the box must be covered). It is low mainly because of
  **tiny text on small images**: on large X-rays recall is 92–97%, on CT 85%, but on 128–256-pixel PET slices only 52% -
  the letters there are 6–8 pixels high. Most of these missed boxes are also unreadable for the OCR after cleaning, which is
  why only 2 words "leak", but a human might still read some of them, so I report both numbers.
- Harmless text that was lost was not hidden for its own sake: it lies inside the safety margin of a PHI line right next to it
  (the synthetic layout puts texts close together).

## Results on the real MIDI-B validation set (headers + pixels, official scorer)

| | Phase 3 | **Phase 4** |
|---|---|---|
| Weighted MIDI-B score | 99.828% | **99.831%** |
| HIPAA checks | 99.992% | **99.996%** |
| Burned-in text checks passed | 6 / 35 | **27 / 35** |
| Images that must stay unchanged, really unchanged | 100% | **99.996%** (1 image of 23,886) |
| Files read by OCR |, | 710 (**3%** of 23,921) |
| Files with hidden text |, | 24, all valid DICOM |

All 10 header lines with the patient's name and dates (*NAME [sex] date / DOB: date*) and most technologist initials are hidden.
The 8 checks that still fail:

| Text | Why it is not hidden |
|---|---|
| *LEITO* (2×) | Portuguese for "bed", a position marker, not an identifier; kept on purpose |
| *LM* | also the standard mammography view "latero-medial"; hiding it would damage every mammogram |
| *ICG* (2×), *ROC* (2×), *YH* | the OCR reads them badly (*"Icg"* with confidence 0.32, *"Roc"* 0.12) or not at all |

The confidence limit for initials is a trade-off: lowering it would also hide the 2 *ICG* images, but OCR noise is read with
similar confidence (*"Lvo"* 0.78, the one image that was changed although it should not have been). I chose the setting that
damages the fewest images and report the misses.

### Pass criteria

| Check | Target | Result |
|---|---|---|
| PHI words still readable after hiding (synthetic test split) | 0 | **2 of 540** |
| PHI box recall (synthetic test split) | 100% | 63% (92–97% on large X-rays, 52% on small PET slices) |
| Harmless text kept | ≥ 90% | 79% (losses are next to hidden PHI) |
| Hidden area smaller than the other methods | yes | **14.7% vs 30.7% / 29.2%** |
| Pixels changed outside the hidden boxes | 0 | **0** |
| MIDI-B burned-in checks | 35 / 35 | 27 / 35 (misses explained above) |
| MIDI-B images that must stay unchanged | 100% | **99.996%** (1 image) |
| Valid DICOM after editing pixels | 100% | **100%** |
| Screening: share of files read | small | **3%** |
| Peak GPU memory | ≤ 8 GB | **5.2 GB** |

## Problems I solved

**1. White text on a bright image was invisible to the OCR.**
My first 8-bit copy of the image cut off the brightest 1% of values; burned-in text is written with the maximum value, so it
merged with bright bone. Reading a second, *bright-text* view (and a *dark-text* view) found all the missed name lines.

**2. OCR splits lines and makes typing mistakes.**
Pieces on the same line are joined before deciding, and the fingerprint match allows small differences (fuzzy matching),
so *"PALDWIN BRANDI"* still matches *BALDWIN BRANDI*.

**3. Ultrasound screens are full of short words.**
My first initials rule hid *LOGIQ, Gn, FR, Map, TRV, AORTA*… on 134 ultrasound images. Now the rule only applies to X-rays,
a code must be one clean word read with good confidence, and codes seen in the images of 3 or more patients are kept.
Over-hiding dropped from 134 images to 1.

**4. Screening must look for PHI, not for "any text".**
With three views, the OCR "sees" some text in 72% of plain CT/MR/PET slices (noise, rulers), but none of it is PHI. Reading a
whole series only when a sample slice has text that *would be hidden* keeps the OCR to 3% of all files.

**5. Small images.**
PET slices of 128 pixels are enlarged before reading. This helps (recall on these images from 55% to 59% in a test), but very
small text stays the main limit of OCR-based cleaning.

**6. Colour images and images without pixels.**
Some colour images missed the *Planar Configuration* field (the default value 0 is now used), and structured reports have
no image at all; they are skipped. One file can never stop the run.

**7. GPU set-up.**
The PyTorch already on the server was built for a newer CUDA than the GPU driver supports, so I installed a matching build
(PyTorch 2.7.1 for CUDA 12.6) in the project environment and used only GPU 0.

## How to reproduce

```bash
CUDA_VISIBLE_DEVICES=0 jupyter nbconvert --to notebook --execute --inplace notebooks/04_burned_in_text.ipynb   # ~1 h
python -m pytest -q tests
```
