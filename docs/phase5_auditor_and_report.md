# Phase 5: Independent auditor, GDPR re-identification tests, and the held-out test set

**Notebook:** [`notebooks/05_auditor_and_report.ipynb`](../notebooks/05_auditor_and_report.ipynb)
**Code:** [`src/fpdeid/auditor.py`](../src/fpdeid/auditor.py) (independent auditor),
[`src/fpdeid/plant_leaks.py`](../src/fpdeid/plant_leaks.py) (planted-leak test),
[`src/fpdeid/reid_attack.py`](../src/fpdeid/reid_attack.py) (re-identification tests),
[`src/fpdeid/pipeline.py`](../src/fpdeid/pipeline.py) + [`scripts/run_split.py`](../scripts/run_split.py) (the full pipeline for one data split)
**Tests:** [`tests/test_auditor.py`](../tests/test_auditor.py)
**Results:** [`results/phase5/`](../results/phase5/)

## What this phase is about

Phases 2–4 built the cleaner and measured it on the validation set. Phase 5 asks three questions a reviewer would ask:

1. **"Can you prove nothing is left?"**: a second, independent program (the *auditor*) checks every output file.
   If it finds something, the pipeline removes it (the *closed loop*).
2. **"Is the shared data really anonymous?"**: I attack my own output the way the EU data-protection authorities
   describe it (*singling out, linkability, inference*), and compare the normal **pseudonymised** output with a
   stricter **anonymised** release.
3. **"Does it work on data you never looked at?"**: the MIDI-B **test set** (322 patients, 29,660 files) is cleaned
   **once**, with the rules frozen before the run, and scored with the official scorer.

## 1. The independent auditor

The auditor does **not** use any code of the cleaner. It has its own, simpler idea of who a patient is:

- From the **original** files of a patient it collects every person name (all name fields), the IDs and accession
  numbers, the dates of birth and of the scans (written in 13 different ways, e.g. *13.06.2017*, *06/13/2017*,
  *13 JUN 2017*), addresses, phone numbers, hospital and station names, device serial numbers, and every original UID.
- In every **output** file it reads every value, also inside sequences, private vendor tags, and bytes that turn out to be
  readable text, and looks for that identity, as whole words only.
- On images it runs **its own OCR** (a normal view and a "bright text" view) on every X-ray, mammogram and ultrasound image,
  and on the middle slice of every CT, MR and PET series.

To avoid false alarms, three general rules were needed (found on the validation set):
the label of a vendor's private block (e.g. *"CTP"*) is not a value; a UID that names a *type* of object (SOP class) is the
same for everyone; and placeholder values (*NONE, UNKNOWN*) and coded vocabulary (*"Long axis"*) identify nobody.

**Closed loop:** in the pipeline, every header hit is removed from the file and written to an audit log
(`audit_auditor_fixes.jsonl`), and the auditor runs a second time to confirm 0 hits.

### Planted leaks: does the auditor really find things?

A checker that never finds anything proves nothing. So I put real identifiers **back** into 120 copies of clean files:
the name in the other order (*"ERIN JONES"*), the surname alone, the patient ID, the accession number, the birth or
scan date in 5 written forms, the doctor, the hospital, and an original UID, in 6 different places (comments, appended to a
description, a private text tag, private bytes, inside a sequence, patient comments). 20 copies got the name or ID
**written into the pixels**.

| Planted in | Planted | Found by the auditor |
|---|---|---|
| headers (names, IDs, dates, places, UIDs) | 100 | **100** |
| pixels (name or ID written into the image) | 20 | **20** |

After the closed-loop fix, the auditor finds **0** of the 100 header leaks again, and all files stay readable.

### On the real outputs

| | Validation | Test |
|---|---|---|
| Elements read | 4,963,487 | 6,689,686 |
| Header hits (fixed by the closed loop) | 0 | 55 |
| Hits per 10,000 elements | 0 | 0.08 |
| Hits after the fix | 0 | **0** |
| Images read by the auditor's OCR / pixel hits | 415 / 0 | 482 / **0** |

The 55 test hits are all of one kind: a patient whose **surname is also an ethnic group word** (e.g. *WHITE*), found in the
*Ethnic Group* field. These are false alarms (the value describes the group, not the person), but removing them is the safe
choice and costs nothing for imaging research.

**What the auditor cannot see:** it knows only the identity written in the patient's own header. The PHI that the official
scorer still finds (section 3) is *not* in the patient's header (other people's names, hospital names and addresses that
appear only in free text), so the auditor could not have found it either. The auditor is a guarantee for the patient's
**own** identity, not for every possible name in the world.

## 2. GDPR re-identification tests

Under GDPR, data is **anonymous** only if a person can no longer be identified "by all the means reasonably likely to be
used" (Recital 26); otherwise it is **pseudonymised** and still personal data (Art. 4(5)). The EU data-protection
authorities (Article 29 Working Party, Opinion 05/2014) name three risks. I measure each one, per patient, with what is
left in the data (the *quasi-identifiers*: sex, age, weight, height, shifted scan dates, modalities, scanner make and model,
body part, number of studies and the gaps between them).

- **Singling out:** is a patient the only one with his/her combination of values? (k-anonymity: *k* = how many patients share it)
- **Linkability:** an **insider** attacker holds the original hospital records of all patients and matches each shared
  patient to the record that agrees on the most values (top-1 / top-5 accuracy; random guessing gives 1/N).
  A **public** attacker only knows sex, age and the year of the scan.
- **Inference:** can the attacker learn a hidden fact, the true year of the scan, the true birth year, the hospital?
  Compared with simply guessing the most common value.

Two kinds of release are compared:

- **Pseudonymised** = the normal FP-DeID output: dates shifted per patient, the key (pseudonym tables) kept by the hospital,
  age, sex, weight and size kept because research needs them.
- **Anonymised** = no key, and the quasi-identifiers made coarser: age in 5-year bands, weight in 10 kg bands, no height,
  no calendar dates at all (only gaps between studies, in months), no scanner model; then **local suppression**: a patient
  who still shares his/her values with fewer than *k* others loses fields one by one (weight first, sex last), and the few
  who are still rare after that are left out of the release.

### Results (test set, 322 patients)

| Release | Unique patients | Min k | Insider link top-1 / top-5 | Public link top-1 | Information kept |
|---|---|---|---|---|---|
| **Pseudonymised (FP-DeID output)** | 99.4% | 1 | **98.4% / 100%** | 28.4% | 94.7% |
| generalised only | 81.1% | 1 | 88.5% / 100% | 12.7% | 60.2% |
| anonymised, k = 2 | 0% | 2 | 31% / 79% | 7% | 39% |
| anonymised, k = 5 | 0% | 5 | **7% / 37%** | 1.7% | 23% |
| anonymised, k = 10 | 0% | 10 | 4% / 22% | 0.9% | 20% |
| *random guess* | | | *0.31% / 1.6%* | *0.31%* | |

*(The validation set, 216 patients, gives the same picture: [`reid_validation.csv`](../results/phase5/reid_validation.csv).)*

| Hidden fact (test) | Attacker, pseudonymised | Attacker, anonymised k=5 | Most-common-value guess |
|---|---|---|---|
| true year of the scan | 13.7% | 12.4% | 12.4% |
| true birth year | 12.7% | 3.1% | 3.1% |
| hospital | 0% | 1.1% | 1.1% |

What this means:

- **The pseudonymised output is personal data, and should be shared as such** (data-sharing agreement, GDPR Art. 89
  safeguards). An insider with the hospital records finds the right patient **98% of the time**, not because a name was
  left, but because the combination *age + weight + scanner + gaps between scans* is unique for almost everyone.
- **The date shift works:** the true scan year cannot be guessed much better than by always guessing the most common year.
  The birth year can (12.7% vs 3.1%), because the exact age is kept.
- **True anonymity has a price.** To make every patient share his/her values with at least 5 others, about three quarters
  of the quasi-identifier values must be generalised or suppressed (information kept 23%). Then the insider's
  success drops to 7% top-1. On a small cohort (322 patients) this is the honest trade-off; on a large data set
  far less would need to be suppressed.
- **Practical recommendation:** share the pseudonymised output for research under an agreement (the default of this
  project); create the anonymised release only for open publication, accepting the loss of age/weight detail.

## 3. The held-out MIDI-B test set (run once, rules frozen)

The rules were frozen (`configs/policy.yaml`, the same code as Phase 4 plus the auditor step) **before** the test set was
cleaned. The test set was cleaned once, scored once, and nothing was changed afterwards.

| | Validation | **Test (once)** |
|---|---|---|
| Patients / files | 216 / 23,921 | 322 / 29,660 |
| **Weighted MIDI-B score** | 99.831% | **99.805%** |
| HIPAA checks (70%) | 99.996% | 99.950% |
| DICOM checks (20%) | 100.0% | 99.995% |
| TCIA checks (10%) | 98.342% | 98.406% |
| UID and patient-ID consistency | 100% | **100%** |
| Burned-in text checks passed | 27 / 35 | **25 / 27** |
| Valid DICOM (dciodvfy, all files) | 100% | **100%** |
| Research fields kept unchanged | 100% | **100%** |
| Audit coverage (every change logged) | 100% | **100%** |
| Original UIDs left | 0 | **0** |
| Time (16 CPU workers + 1 GPU) | 28 min | 37 min |

The test score is only **0.03 points** below validation: the rules were not over-fitted to the validation patients.
For reference, the ten teams that finished the MIDI-B challenge (MICCAI 2024) scored 97.91–99.93% on this test set.

### Error analysis: every failed check, by cause

| Cause | Checks | Privacy problem? |
|---|---|---|
| private value empty in the input (the scorer expects an empty value we do not write) | 43,917 | no |
| private tag inside a private sequence: the scorer cannot read it (fails even for unchanged files) | 14,914 | no |
| private value that TCIA's vendor knowledge base removes, but holds no PHI (we keep it) | 12,842 | no |
| icon image removed on purpose (a small copy of the image can show burned-in text) | 6,097 | no |
| **leak: identifier in free text not found** | 831 | **yes** |
| label of a private block whose content was binary and removed | 409 | no |
| **leak: date not shifted (private "start time" field)** | 90 | **yes** |
| other utility checks (values kept/changed slightly differently than the key) | 432 | no |
| burned-in text not hidden / pixels changed although not needed | 2 / 9 | yes / no |

Of the 13,765 failed privacy checks, 12,842 are vendor private values that TCIA's knowledge base removes but that hold no
PHI (kept on purpose, as in Phase 3). The **921 real leaks** (0.10% of 884,700 privacy checks) come from a few kinds of values:

| What is left | Checks | Why the pipeline missed it |
|---|---|---|
| a hospital name, e.g. *"Admitted to White Hospital"*, *"POST CONTRAST at Meyer, Moore and Coffey Community Hospital"* | 497 | it is not the patient's own hospital in the header, so it is not in the fingerprint |
| a phone number, e.g. *"Patient Followup request for, call 7404159161"* | 135 | the doctor's name was removed, but 10 digits without any separator are not one of my phone patterns |
| *"1348576679.0"* in a private *"Start time (secs)"* field | 90 | a Unix time written as a decimal number; my time-stamp rule only recognises whole numbers |
| other parts of names that are not in the header, removed only partly | 76 | the fingerprint knows only the patient's own identity |
| Referenced File IDs made of numbers | 64 | file names that the answer key treats as identifiers |
| a street address, e.g. *"2925 N Ave Sioux Falls, SD 57104"* | 45 | the name next to it was removed, but an address that is not in the header stays (my address pattern needs a street type) |
| *RD, CH, CW* in structured reports | 14 | 2-letter staff initials; kept because 2-letter codes are very common in reports |

These are honest misses of rules I designed on the validation set, where they did not occur (or only rarely).
I list them here instead of changing the rules after seeing the test set: changing them now would make the test score
meaningless. Each one has a clear, simple fix for a next version (10-digit phone numbers next to "call/phone",
decimal Unix times, generic "... Hospital/Clinic" names and US addresses), which must then be checked on the validation
set and on new data, not on this test set.

The 2 burned-in text checks that fail are short staff initials that the OCR read with low confidence (the same limit
as in Phase 4).

## 4. Privacy vs. utility

![Privacy vs utility](../results/phase5/privacy_utility.png)

**Left:** every method of the project on the validation set, with the official scorer (privacy = identifier checks
passed, utility = "keep this" checks passed). The standard Basic Profile (Baselines A and B) is private but throws away a
third of the useful data; the rules engine (Phase 2) moves far to the right; the fingerprint (Phase 3) reaches the top
right corner; NER is clearly worse on both axes. The test-set point (red star) lies next to the validation point.
**Right:** the GDPR view of a release: going from the pseudonymised output to stronger k-anonymity, privacy against an
insider rises from 2% to 96%, while the information kept in the quasi-identifiers falls from 95% to 20%.

## 5. Ablation table: what each part adds (validation, official scorer)

| Configuration | Weighted score | Privacy checks failed | Utility checks failed |
|---|---|---|---|
| Baseline A: PS3.15 Basic Profile | 95.31% | 21,358 | 1,670,457 |
| Baseline B: + options (dates, clean descriptors) | 95.87% | 31,131 | 1,458,139 |
| Phase 2: policy rules engine | 99.15% | 4,577 | 302,210 |
| Phase 3: rules + generic patterns only | 97.41% | 25,428 | 54,997 |
| Phase 3: rules + NER (spaCy) + patterns | 98.80% | 11,119 | 163,270 |
| Phase 3: rules + fingerprint, exact values | 96.65% | 47,854 | 54,999 |
| Phase 3: rules + fingerprint with variants | 99.08% | 19,764 | 54,999 |
| **Phase 3: FP-DeID text (fingerprint + patterns + guard)** | **99.83%** | **5,198** | **55,003** |
| Phase 4: + burned-in text (value-aware OCR) | 99.831% | 5,177 | 55,004 |
| Phase 5: + closed-loop auditor | 99.831% | 5,177 | 55,004 |
| **Test set (once): full FP-DeID** | **99.805%** | 13,765 | 65,778 |

The novelty is defended on three points that the plan set before the work started:
1. the fingerprint rows leak **53% less** than NER with the same rules and patterns, and remove **66% less** useful data;
2. the value-aware parts (private tags by value type, cleaning text word by word, OCR that hides only identifying lines)
   cut over-cleaning from 1.46 million to 55 thousand checks without adding leaks;
3. the independent auditor finds **100% of the planted leaks** (100 in headers, 20 in pixels), and leaves 0 hits in the
   final output. On validation it found nothing to fix, the cleaner had already removed the patient's own identity
   everywhere; on the test set it removed 55 values.

## Pass criteria

| Check | Target | Result |
|---|---|---|
| MIDI-B score on the **test** set | ≥ 99% | **99.805%** |
| Validation score with the frozen pipeline | = Phase 4 | **99.831%** (identical, reproducible) |
| Leaks in text and pixels on the test set | 0, or each listed | 921 text + 2 pixel checks, all listed above with a cause |
| Auditor finds planted leaks | 100% | headers **100 / 100**, pixels **20 / 20** |
| Auditor false alarms | reported per 10,000 elements | 0 (validation), 0.08 (test) |
| Auditor hits left after the closed loop | 0 | **0** |
| Singling out (unique %, min k) | reported | pseudonymised 99.4% unique; anonymised k=5: 0%, min k = 5 |
| Linkability vs random | reported | 98.4% (pseudonymised), 7% (anonymised k=5) vs 0.31% |
| Inference vs majority guess | reported | scan year 13.7% vs 12.4%; birth year 12.7% vs 3.1% |
| Research-field retention on test | 100% | **100%** |
| DICOM validity on test (all 29,660 files) | 100% | **100%** |
| Audit coverage on test | 100% | **100%** |

## Problems I solved

**1. An auditor that shares the cleaner's code would share its blind spots.**
So the auditor is a separate module with its own, simpler identity collection, its own matching and its own OCR. The only
thing both have in common is the input data.

**2. The first version of the auditor raised false alarms.**
On the validation set it flagged the label of a private block (*"CTP"*), a Siemens SOP Class UID, the placeholder
*"NONE"* and the code meaning *"Long axis"* (a patient was called *Long*). Four general rules (private block labels, class
UIDs, placeholders, coded vocabulary) removed all of them without lowering the planted-leak recall.

**3. Proving that a checker works.**
"0 hits" can also mean "the checker is blind". The planted-leak test gives the auditor a known answer: 120 leaks in
different forms and places. Month names in dates (*13 JUN 2017*) were added after I first wrote the test. Writing the
planted text into 16-bit images also needed care: the drawing library only draws on 8-bit images, so the text is drawn
on an 8-bit mask first and then written into the real pixels with the image's brightest value.

**4. Attacking with the shifted year makes the attack weaker.**
Because dates are shifted by 1–10 years, an attacker who compares scan years matches the wrong patients. A smart attacker
ignores the year, so the "insider" attack uses every value except the calendar year, the strongest attack I could build.

**5. k-anonymity on a small cohort.**
With 216–322 patients, even *sex + 5-year age band + modality* leaves about 40% of the patients unique, so generalising
alone cannot reach k = 5. I added local suppression (blank a field only for the patients who are still rare, repeated until
nothing changes) and, for the very few who stay rare, leaving them out of the release. The loss of information is reported
next to every privacy number, so the trade-off is visible.

**6. One test-set run only.**
The whole pipeline is one function per data split (`pipeline.run_split` + the auditor step + the official scorer), so the
validation run and the test run use exactly the same code. The validation run reproduces the Phase 4 score exactly, and the
test set was run once.

## How to reproduce

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_split.py validation     # ~35 min (headers + OCR + auditor + scorer)
CUDA_VISIBLE_DEVICES=0 python scripts/run_split.py test           # ~45 min, run once
CUDA_VISIBLE_DEVICES=0 jupyter nbconvert --to notebook --execute --inplace notebooks/05_auditor_and_report.ipynb
python -m pytest -q tests
```
