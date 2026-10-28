# Phase 2: Policy rules engine and audit log

Code: [`engine.py`](../src/fpdeid/engine.py), [`checks.py`](../src/fpdeid/checks.py) · policy at the end of this phase:
[`policy_phase2.yaml`](../configs/policy_phase2.yaml) · tests: [`test_engine.py`](../tests/test_engine.py)

The policy is a YAML file that a data protection officer could read. The engine applies it to every element of every
file, including elements inside sequences and private blocks, and writes each change to `audit.jsonl` with the reason,
the PS3.15 option it implements and a GDPR label. Old values are stored only as a short hash.

What the policy does, compared with BB:

- starts from PS3.15 Table E.1-1 with *Retain Patient Characteristics* and *Retain Longitudinal Temporal Information with
  Modified Dates* (one shift per patient, so intervals survive);
- replaces every UID that appears in the key table, wherever it is, not only in the UID fields;
- decides private tags by value representation: numbers and codes (DS, IS, FL, US, CS, ...) stay, dates are shifted, UIDs
  pseudonymised, text and binary removed for now; the Safe Private Attributes table (E.3.10-1) is always kept;
- shifts numbers that are really dates (`SeriesNumber = 20160807`) and integers in fields named *time/date/stamp* that hold a
  Unix time between 1990 and 2030;
- drops serial numbers, station names and AE titles (*Retain Device Identity* caused about 10,000 leaks in BB) but keeps
  manufacturer and model;
- adds required Type 2 elements that were missing in the input, and fills a few Type 1 values that follow from the SOP
  class (`FOR PRESENTATION`, laterality `U`).

GDPR labels: removal is data minimisation (Art. 5(1)(c)); pseudonyms and shifted dates are pseudonymisation (Art. 4(5)) and
the output stays personal data because the key exists (Recital 26); kept fields fall under purpose limitation (Art. 5(1)(b)).

## Results on validation

| | BA | BB | Phase 2 |
|---|---|---|---|
| Weighted score | 95.31% | 95.87% | 99.15% |
| HIPAA | 99.989% | 99.988% | 99.994% |
| DICOM | 99.885% | 99.885% | 100.000% |
| TCIA best practice | 53.42% | 59.01% | 91.54% |
| Privacy checks failed | 21,358 | 31,131 | 4,577 |
| Utility checks failed | 1,670,457 | 1,458,139 | 302,210 |

![Phase 2 vs baselines](../results/phase2/scores_vs_baselines.png)

Date shifts, UID changes, UID and Patient ID consistency and pixel preservation pass 100%. Independently of the scorer,
[`checks.py`](../src/fpdeid/checks.py) diffs every input/output pair: all 1,349,623 changes are in the audit log, all 42
research fields in [`utility_tags.yaml`](../configs/utility_tags.yaml) are unchanged, date intervals are intact, no original
UID is left, and the 216 / 241 / 280 / 23,921 patients, studies, series and instances are still there. Two runs give
byte-identical files. Speed: 1,354 files/s with 16 processes.

The value-type rule for private tags did most of the work here: TCIA best practice went from 59% to over 91%. I could not
use TCIA's own private-tag knowledge base (it is not published, and reading it out of the answer key would be cheating).

## What still fails

The target of 99.5% on keep-checks is missed (94.19%). Almost all of it is text: 235,125 private text values and 63,296
free-text fields that the engine removes because they might hold a name. That needs the fingerprint (Phase 3).
4,519 failures are private numbers that TCIA's knowledge base removes although they are all `0`; I have no general rule for
those. 1,763 are icon images that I remove on purpose because a thumbnail can show burned-in text.

## Bugs found while building it

- The first full run left 236 UIDs inside structured-report content unchanged: sequences not listed in the PS3.15 table
  were kept untouched. Since then every kept sequence is cleaned inside.
- Private creators: a block label is kept only if some element of its block survives.
- One GE private field stores the acquisition time in Unix seconds (`1228050572`). Without the field-name condition
  the timestamp rule would also have shifted PET event counts.
- `dciodvfy` messages contain UIDs, so after remapping every old error looked new. The validity check now masks UIDs.
