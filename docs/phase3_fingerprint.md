# Phase 3: The PHI fingerprint (free text and private tags)

**Notebook:** [`notebooks/03_fingerprint_text_private.ipynb`](../notebooks/03_fingerprint_text_private.ipynb)
**Code:** [`src/fpdeid/fingerprint.py`](../src/fpdeid/fingerprint.py), [`src/fpdeid/engine.py`](../src/fpdeid/engine.py),
[`src/fpdeid/ner_baseline.py`](../src/fpdeid/ner_baseline.py) (comparison method)
**Policy:** [`configs/policy.yaml`](../configs/policy.yaml) (Phase 2 version kept as [`configs/policy_phase2.yaml`](../configs/policy_phase2.yaml))
**Tests:** [`tests/test_fingerprint.py`](../tests/test_fingerprint.py)
**Results:** [`results/phase3/`](../results/phase3/)

## What this phase is about

After Phase 2, the biggest loss of useful data was **text**: study and series descriptions, protocol names,
comments and vendor text fields. They were deleted completely because they *sometimes* contain the patient's name.
But most of the text is useful, *"PET-CT SERIES for Amy Wilson"* should become *"PET-CT SERIES"*, not nothing.

To do that safely I need to know **which words are the patient**. Existing tools guess this with a language model
(named-entity recognition, NER). My idea is simpler and more reliable:

> **The best detector of a patient's leaked identity is the patient's own identity.**

Before cleaning, I read every patient's identifiers from the fields where identity is *known* to live
(name, IDs, birth and study dates, address, phone, institution, device names). I call this set the patient's
**fingerprint**. Then I search for the fingerprint in every text value and remove only the matching words.

## How the fingerprint works

**1. Collect** the identifiers of each patient from all of their files (one fast pass over the headers: 14 s for 216 patients).

**2. Expand every value into all the ways it can be written in text:**

| Value in its home field | Variants searched for in text |
|---|---|
| `WILSON^AMY` (Patient Name) | *Amy Wilson*, *Wilson Amy*, *A Wilson*, *WILSON*, *AMY* (any case, any separator) |
| `19811129` (a date) | *11/29/1981*, *29.11.1981*, *1981-11-29*, *November 29, 1981*, *29-Nov-1981* ... (19 formats) |
| `689-96-0899` (Other Patient IDs) | *689-96-0899*, *689960899* |
| `Compton, Holmes and Jones Hospital` | the full name and its acronym **CHAJH** |
| `4643 Stern Dr Merced, CA 95348` | the full address and its parts |

The acronym was a real finding: hospitals were written as *"PET-CT STUDY at CHAJH"* in the descriptions.

**3. Backup patterns** for identifiers that are in no home field: social-security numbers, phone numbers (also with
extensions), e-mail addresses, 8-digit and slash dates, *"Dr. X"*, staff initials after *"by"*, long numbers after
*"for / MRN / ID / #"*, postal addresses (*"..., AR 71857"*) and site codes at the end of a description (*"... at FMCC"*).
Each new pattern was first checked on the validation answer key: it may only remove text that the key marks as PHI.

**4. A word guard.** A single word (for example a surname) is never removed on its own if it appears in the text of
many *other* patients, then it is an ordinary word (a patient called "Brain" must not erase "brain").
499 such words were found (ABDOMEN, AXIAL, ADULT, ...).

**5. Explainable.** Every removal in the audit log says which entry matched, for example
`removed: name (PatientName), institution (InstitutionName)`.

The engine now **cleans** descriptions (PS3.15 *Clean Descriptors* option) and private text tags instead of deleting
them, and every other kept text value also goes through the fingerprint as a safety net. Research-critical fields and
coded values are never touched.

## Results (MIDI-B validation set, 23,921 files, official scorer)

| | BB (best baseline) | Phase 2 | **Phase 3** |
|---|---|---|---|
| **Weighted MIDI-B score** | 95.87% | 99.15% | **99.83%** |
| HIPAA checks | 99.988% | 99.994% | **99.992%** |
| DICOM checks | 99.885% | 100% | **100%** |
| TCIA best-practice checks | 59.0% | 91.5% | **98.3%** |
| Utility checks failed (over-cleaning) | 1,458,139 | 302,210 | **55,003** |
| Privacy checks failed | 31,131 | 4,577 | 5,198 (see below) |

For reference, the ten teams that finished the MIDI-B challenge in 2024 scored 97.91–99.93% (on the test set).

### Ablation study, is it really the fingerprint that helps?

Every row is a full run on the validation set, scored with the official script. "Text leaks" and "useful text removed"
count only the free-text and private text fields, where the methods differ.

| Method | Weighted score | Text leaks | Useful text removed | Time (s) |
|---|---|---|---|---|
| 1. Rules only (Phase 2), text deleted | 99.150% | 4,548 | 252,630 |, |
| 2. Rules + generic patterns only | 97.406% | 25,399 | 53,026 | 20 |
| 3. **Rules + NER (spaCy) + the same patterns** | 98.799% | 11,090 | 126,512 | 29 |
| 4. Fingerprint with exact values only | 96.651% | 47,825 | 53,032 | 29 |
| 5. Fingerprint with variants, no patterns | 99.077% | 19,735 | 53,032 | 29 |
| **6. FP-DeID (fingerprint + variants + patterns + guard)** | **99.828%** | **5,169** | **53,032** | 31 |
| 7. FP-DeID without the word guard | 99.828% | 5,169 | 53,032 | 28 |
| 8. FP-DeID with a site-level fingerprint (all patients) | 99.828% | 5,169 | 53,032 | 56 |

![Ablation](../results/phase3/ablation.png)

What this shows:
- **Fingerprint vs. NER:** FP-DeID leaks **53% less** (5,169 vs 11,090) and removes **58% less useful text**
  (53,032 vs 126,512). NER guesses: it misses unusual names and removes ordinary words (it even tags *"PET-CT"* as an
  organisation). The fingerprint knows the exact values.
- **The variants matter:** exact values alone miss most of the PHI (47,825 leaks); writing names, dates and IDs in all
  their forms cuts that to 19,735, and the backup patterns to 5,169.
- **The word guard** made no difference on this data (no patient's name is also an ordinary word here), but it costs
  nothing and protects against that case.
- **A site-level fingerprint** (adding every other patient's names and addresses) gave no gain on MIDI-B and doubled the
  time, so it is available as an option but switched off.

Of the 5,169 remaining "text leaks", 5,130 are private number fields that TCIA's vendor knowledge base removes
(all of them hold `0`, they are not PHI). **Only 39 are real identifiers left in text** (listed in
[`results/phase3/leak_list.csv`](../results/phase3/leak_list.csv)): 22 short codes that are the whole content of a
structured-report text field (mostly staff initials such as *"KM"*), 16 bare 10-digit numbers in patient comments and
1 other. They are left on purpose: a rule that removes every 2-letter text or every 10-digit number would destroy more
useful data than it protects.

### Pass criteria

| Check | Target | Result |
|---|---|---|
| MIDI-B weighted score (validation) | ≥ 99% | **99.83%** |
| Residual fingerprint hits in the output (independent re-scan) | 0 | **0** |
| Leaks in free text / private text | 0, or each listed with a cause | **39 real leaks, all listed** |
| Fingerprint vs NER | fewer leaks **and** less useful text removed | **5,169 vs 11,090** and **53,032 vs 126,512** |
| Removals explained by a fingerprint entry or pattern | 100% | **100%** |
| Extra time per file | ≤ 20 ms | **≈ 3–8 ms** |
| Valid DICOM / audit coverage / research fields kept | 100% | **100% / 100% / 100%** |

### What is still failing, and why

| Cause | Failed checks | Note |
|---|---|---|
| Private value empty in the input | 39,338 | the same checks fail for completely unchanged files |
| Private tag inside a private sequence | 13,686 | the scorer cannot read these; they also fail for unchanged files |
| Private number fields removed by TCIA's vendor knowledge base | 5,130 | not PHI (all `0`); would need a vendor-specific list |
| Icon images removed on purpose | 1,763 | thumbnails can show burned-in text |
| Real identifiers left in text | 39 | listed above (SR initials, bare 10-digit numbers) |
| Burned-in text in pixels | 29 | Phase 4 |

## Problems I solved

**1. Finding the patient's name without guessing it.**
Instead of a language model, the fingerprint uses the patient's own values, written in every common form.
This is faster, needs no training data, works for unusual names, and every removal can be explained.

**2. Hospitals hidden as acronyms.**
Descriptions often ended with *"at CHAJH"* or *"at MSCH"*. These are the initials of the patient's own institution
(*Compton, Holmes and Jones Hospital*), so the fingerprint also contains the acronym of every institution name.
For site codes that are in no field at all, a pattern removes a short capital-letter code after *"at"* at the end of a description.

**3. Ordinary words must survive.**
A data-driven guard: a word that appears in the text of more than 2% of other patients is not used on its own.

**4. Context words left behind.**
After removing a surname, *"Dr."* was left alone (*"Ordered for by Dr."*). The doctor pattern now runs before the
fingerprint, so *"Dr. Torres"* is removed as a whole; a small tidy step removes leftover *":"*, *"for"* and extra spaces.
The doctor pattern also needs the dot, so the street *"Stern Dr Merced"* is not mistaken for a doctor.

**5. Vendor fields with no dictionary.**
Some private values have an unknown type (UN). If their bytes are readable text, they are cleaned like text;
if not, they are removed. Private text fields whose vendor name says they are an identifier
(*suite id, service id, serial, station, coil id, ...*) are removed instead of cleaned.

**6. Empty private blocks.**
Some private blocks contain only a vendor label and no data already in the input. My first version removed such labels
(about 45,000 failed checks); now a label is removed only if *my cleaning* emptied its block.

**7. The cleaner must not touch research data or codes.**
In 143 files the model name was also written as a "device ID", so the cleaner removed the model name. Now research-critical
fields and coded values are never cleaned, and a "device ID" that is just the product (manufacturer, model, software) name
is not treated as an identifier.

**8. Mixing standard options correctly.**
The PS3.15 table uses "C" for different things in different options. Only "C" from the *Clean Descriptors* option means
"clean the text"; "C" from the dates option on a non-date field (for example *Timezone Offset*) means remove.

**9. A test that would be unfair otherwise.**
The comparison with NER uses the same engine, the same rules and the same backup patterns; only the way of finding PHI
in text changes. The NER model is loaded once per process and its results are cached, so it is not slowed down unfairly.

## How to reproduce

```bash
python scripts/dev_eval.py my_run          # quick loop on dev-mini (~5 min)
jupyter nbconvert --to notebook --execute --inplace notebooks/03_fingerprint_text_private.ipynb   # full ablation (~1.5 h)
python -m pytest -q tests
```

The test set is run once, with the final version, when its answer key is available (see Phase 5).
