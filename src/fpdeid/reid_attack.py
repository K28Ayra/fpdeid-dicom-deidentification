"""Phase 5: GDPR re-identification tests on the de-identified data (patient level).

An attacker sees what is left in the shared data (the quasi-identifiers, QIs): sex, age, weight, height,
the (shifted) study dates, modalities, scanner make and model, body parts. Three questions (the three
risks named by the EU Article 29 Working Party, Opinion 05/2014):

- singling out : is a patient the only one with his/her combination of QIs?  (k-anonymity)
- linkability  : with the original hospital records in hand, can the attacker find which record
                 belongs to a shared patient?  (top-1 / top-5 linking accuracy vs. random guessing)
- inference    : can the attacker learn a hidden fact (true study year, true birth year, hospital)?

All tables are one row per patient. Headers are read from one file per series (enough for these fields).
"""

import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom

QI_COLUMNS = ["sex", "age", "weight", "height", "modalities", "manufacturers", "models", "body_parts",
              "n_studies", "study_gaps", "study_year"]


# ---------------------------------------------------------------
# 1. Patient tables
# ---------------------------------------------------------------

def age_in_years(text):
    """'037Y' -> 37, '006M' -> 0, '' -> None."""
    text = str(text).strip().upper()
    if len(text) < 2 or not text[:-1].isdigit():
        return None
    number = int(text[:-1])
    unit = text[-1]
    if unit == "Y":
        return number
    return 0


def number_or_none(value, digits):
    try:
        number = round(float(value), digits)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def to_date(text):
    try:
        return datetime.datetime.strptime(str(text).strip()[:8], "%Y%m%d")
    except ValueError:
        return None


def read_series_header(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    row = {
        "patient": str(ds.get("PatientID", "")),
        "sex": str(ds.get("PatientSex", "")).strip(),
        "age": age_in_years(ds.get("PatientAge", "")),
        "weight": number_or_none(ds.get("PatientWeight", None), 1),
        "height": number_or_none(ds.get("PatientSize", None), 2),
        "modality": str(ds.get("Modality", "")),
        "manufacturer": str(ds.get("Manufacturer", "")).strip().upper(),
        "model": str(ds.get("ManufacturerModelName", "")).strip().upper(),
        "body_part": str(ds.get("BodyPartExamined", "")).strip().upper(),
        "study_uid": str(ds.get("StudyInstanceUID", "")),
        "study_date": str(ds.get("StudyDate", "")),
        "birth_date": str(ds.get("PatientBirthDate", "")),
        "institution": str(ds.get("InstitutionName", "")).strip().upper(),
    }
    return row


def first_value(values):
    for value in values:
        if value is not None and value == value and value != "":
            return value
    return None


def joined(values):
    """Sorted, unique, non-empty values as one text: ['MR', 'CT', 'MR'] -> 'CT|MR'."""
    clean = set()
    for value in values:
        if value is not None and value == value and value != "":
            clean.add(str(value))
    return "|".join(sorted(clean))


def patient_table(series_paths, workers=16):
    """series_paths: one file path per series -> one row per patient with the QIs."""
    with Pool(workers) as pool:
        rows = pool.map(read_series_header, series_paths, chunksize=20)
    series = pd.DataFrame(rows)

    patients = []
    for patient, group in series.groupby("patient"):
        study_dates = {}
        for _, row in group.iterrows():
            date = to_date(row["study_date"])
            if date is not None:
                study_dates[row["study_uid"]] = date
        dates = sorted(set(study_dates.values()))
        gaps = []
        for i in range(1, len(dates)):
            gaps.append(str((dates[i] - dates[i - 1]).days))
        first_date = dates[0] if len(dates) > 0 else None

        record = {
            "patient": patient,
            "sex": first_value(group["sex"]),
            "age": first_value(group["age"]),
            "weight": first_value(group["weight"]),
            "height": first_value(group["height"]),
            "modalities": joined(group["modality"]),
            "manufacturers": joined(group["manufacturer"]),
            "models": joined(group["model"]),
            "body_parts": joined(group["body_part"]),
            "n_studies": group["study_uid"].nunique(),
            "study_gaps": "|".join(gaps),
            "first_study_date": first_date,
            "study_year": first_date.year if first_date is not None else None,
            "birth_year": first_value([d[:4] for d in group["birth_date"] if len(d) >= 4]),
            "institution": joined(group["institution"]),
        }
        patients.append(record)
    return pd.DataFrame(patients)


# ---------------------------------------------------------------
# 2. The "anonymised" release: generalise the QIs further
# ---------------------------------------------------------------

def generalise(table):
    """Coarser QIs, the same for every patient.

    - age in 5-year bands (90+ together), weight in 10 kg bands, height removed
    - dates: no calendar date at all (no study year); only the gaps between studies, rounded to months
    - scanner model removed (manufacturer kept)
    """
    out = table.copy()
    bands = []
    for age in out["age"]:
        if age is None or age != age:
            bands.append(None)
        elif age >= 90:
            bands.append("90+")
        else:
            low = int(age // 5) * 5
            bands.append(str(low) + "-" + str(low + 4))
    out["age"] = bands
    weights = []
    for weight in out["weight"]:
        if weight is None or weight != weight:
            weights.append(None)
        else:
            low = int(weight // 10) * 10
            weights.append(str(low) + "-" + str(low + 9))
    out["weight"] = weights
    out["height"] = None
    out["study_year"] = None
    out["first_study_date"] = None
    months = []
    for gaps in out["study_gaps"]:
        if gaps == "":
            months.append("")
        else:
            rounded = []
            for gap in gaps.split("|"):
                rounded.append(str(int(round(int(gap) / 30.4))))
            months.append("|".join(rounded))
    out["study_gaps"] = months
    out["models"] = ""
    return out


SUPPRESS_ORDER = ["weight", "study_gaps", "manufacturers", "body_parts", "n_studies", "age", "modalities", "sex"]
ANONYMISED_QIS = ["sex", "age", "weight", "modalities", "manufacturers", "body_parts", "n_studies", "study_gaps"]


def anonymise(table, k=5):
    """Pseudonymised -> anonymised release: generalise, then suppress rare values (k-anonymity).

    After generalising, a patient whose combination of QIs is shared by fewer than k patients loses
    one field (replaced by "*"), in the order of SUPPRESS_ORDER (the most identifying first),
    until every patient has k >= k. Patients who are still rare after that are left out of the release.
    Returns (anonymised table, share of QI values suppressed).
    """
    out = generalise(table)
    columns = ANONYMISED_QIS
    out[columns] = out[columns].astype(object)
    # suppressing some patients can leave their old group with fewer than k, so repeat until nothing changes
    for repeat in range(len(SUPPRESS_ORDER)):
        changed = False
        for column in SUPPRESS_ORDER:
            rare = k_values(out, columns) < k
            newly = rare & (out[column] != "*")
            if newly.sum() > 0:
                out.loc[newly, column] = "*"
                changed = True
        if not changed:
            break
    suppressed = (out[columns] == "*").to_numpy().mean()
    # the few patients who are still in a group smaller than k are withheld from the release
    still_rare = k_values(out, columns) < k
    out = out[~still_rare].reset_index(drop=True)
    return out, round(100 * suppressed, 1)


# ---------------------------------------------------------------
# 3. Singling out (k-anonymity)
# ---------------------------------------------------------------

def k_values(table, columns):
    """For each patient: how many patients share exactly the same values in `columns` (k)."""
    keys = table[columns].astype(str).agg(" / ".join, axis=1)
    counts = keys.map(keys.value_counts())
    return counts


def singling_out(table, columns):
    k = k_values(table, columns)
    result = {
        "patients": len(table),
        "unique (k=1) %": round(100 * (k == 1).mean(), 1),
        "k < 5 %": round(100 * (k < 5).mean(), 1),
        "min k": int(k.min()),
        "median k": float(k.median()),
    }
    return result


# ---------------------------------------------------------------
# 4. Linkability: match each shared patient to the attacker's original records
# ---------------------------------------------------------------

def same(a, b):
    """1 if both values are known and equal, else 0."""
    if a is None or b is None or a != a or b != b or a == "" or b == "" or a == "*" or b == "*":
        return 0
    if a == b:
        return 1
    return 0


def match_score(shared, attacker_view, columns):
    """How many QIs of the shared patient agree with one original record (after the attacker's own
    generalisation of the record, so both sides are written the same way)."""
    score = 0
    for column in columns:
        score = score + same(shared[column], attacker_view[column])
    return score


def chance_in_top(scores, true_index, top):
    """Probability that the true record is in the attacker's top-n list (ties are broken at random)."""
    true_score = scores[true_index]
    better = int((scores > true_score).sum())
    tied = int((scores == true_score).sum())     # includes the true record
    places_left = top - better
    if places_left <= 0:
        return 0.0
    return min(1.0, places_left / tied)


def linkability(shared_table, original_table, link, columns, prepare=None):
    """shared_table: output patients; original_table: attacker's records; link: {shared id: original id}.

    prepare(table) is applied to the attacker's records so that they look like the shared data
    (e.g. the same age bands). Returns top-1 / top-5 accuracy and the random baseline.
    """
    attacker = original_table.copy()
    if prepare is not None:
        attacker = prepare(attacker)
    attacker = attacker.reset_index(drop=True)
    position = {pid: i for i, pid in enumerate(attacker["patient"])}

    top1 = []
    top5 = []
    attacker_rows = attacker.to_dict("records")
    for shared in shared_table.to_dict("records"):
        true_id = link.get(shared["patient"])
        if true_id not in position:
            continue
        scores = []
        for record in attacker_rows:
            scores.append(match_score(shared, record, columns))
        scores = np.array(scores)
        top1.append(chance_in_top(scores, position[true_id], 1))
        top5.append(chance_in_top(scores, position[true_id], 5))
    n = len(attacker)
    result = {
        "patients": len(top1),
        "top-1 %": round(100 * float(np.mean(top1)), 1),
        "top-5 %": round(100 * float(np.mean(top5)), 1),
        "random top-1 %": round(100 / n, 2),
        "random top-5 %": round(100 * 5 / n, 2),
    }
    return result


# ---------------------------------------------------------------
# 5. Inference: guess a hidden fact
# ---------------------------------------------------------------

def accuracy(guesses, truths):
    pairs = [(g, t) for g, t in zip(guesses, truths) if t is not None and t == t and t != ""]
    if len(pairs) == 0:
        return None
    right = sum(1 for g, t in pairs if str(g) == str(t))
    return round(100 * right / len(pairs), 1)


def majority_guess(truths):
    clean = [str(t) for t in truths if t is not None and t == t and t != ""]
    if len(clean) == 0:
        return None
    return pd.Series(clean).value_counts().index[0]


def inference(shared_table, original_table, link, shift_years_guess=5):
    """Three hidden facts the attacker tries to learn about each shared patient.

    - true study year : guess = shared (shifted) year + the middle of the published shift range
    - true birth year : guess = guessed study year - age
    - hospital name   : guess = the hospital most often seen with the same scanner models in the
                        attacker's background records (all other patients, leave-one-out)
    Each is compared with the majority guess (always the most common true value).
    """
    original_by_id = original_table.set_index("patient")
    rows = []
    for shared in shared_table.to_dict("records"):
        true_id = link.get(shared["patient"])
        if true_id not in original_by_id.index:
            continue
        truth = original_by_id.loc[true_id]
        row = {"true study year": truth["study_year"], "true birth year": truth["birth_year"],
               "true hospital": truth["institution"]}
        guess_year = None
        if shared["study_year"] is not None and shared["study_year"] == shared["study_year"] \
                and shared["study_year"] != "*":
            guess_year = int(shared["study_year"]) + shift_years_guess
        row["guess study year"] = guess_year
        guess_birth = None
        if guess_year is not None and shared["age"] is not None and shared["age"] == shared["age"] \
                and isinstance(shared["age"], (int, float)):
            guess_birth = str(int(guess_year - shared["age"]))
        row["guess birth year"] = guess_birth
        others = original_table[(original_table["patient"] != true_id) &
                                (original_table["models"] == shared["models"]) & (original_table["models"] != "")]
        row["guess hospital"] = majority_guess(others["institution"])     # None if no other patient has these models
        rows.append(row)
    table = pd.DataFrame(rows)

    results = []
    for fact in ["study year", "birth year", "hospital"]:
        truths = table["true " + fact].tolist()
        majority = majority_guess(truths)
        # when the shared data gives no clue, the attacker falls back to the majority guess
        guesses = []
        for guess in table["guess " + fact]:
            if guess is None or guess != guess:
                guesses.append(majority)
            else:
                guesses.append(guess)
        results.append({
            "hidden fact": fact,
            "attacker accuracy %": accuracy(guesses, truths),
            "majority-guess accuracy %": accuracy([majority] * len(truths), truths),
            "patients with this fact": sum(1 for t in truths if t is not None and t == t and t != ""),
        })
    return pd.DataFrame(results)


def information_kept(original_table, released_table, link, columns):
    """A simple utility score for a release: per QI value 1 = exact, 0.5 = coarser or shifted, 0 = removed."""
    original_by_id = original_table.set_index("patient")
    released_by_id = released_table.set_index("patient")
    points = 0.0
    count = 0
    for shared_id, true_id in link.items():
        if true_id not in original_by_id.index:
            continue
        original = original_by_id.loc[true_id]
        withheld = shared_id not in released_by_id.index      # a withheld patient keeps nothing
        for column in columns:
            before = original[column]
            if before is None or before != before or before == "":
                continue
            if withheld:
                count = count + 1
                continue
            after = released_by_id.loc[shared_id, column]
            count = count + 1
            if after is None or after != after or after == "" or after == "*":
                points = points + 0.0
            elif str(after) == str(before):
                points = points + 1.0
            else:
                points = points + 0.5
    return round(100 * points / count, 1)
