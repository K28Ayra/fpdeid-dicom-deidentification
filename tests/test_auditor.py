"""Tests for the Phase 5 independent auditor and the re-identification attack (no GPU needed)."""

import pandas as pd
from pydicom.dataset import Dataset

from fpdeid import auditor
from fpdeid import reid_attack


def make_original():
    ds = Dataset()
    ds.PatientName = "JONES^ERIN"
    ds.PatientID = "4107422692"
    ds.PatientBirthDate = "19371112"
    ds.StudyInstanceUID = "1.2.3.4.5.6"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    return ds


def matchers_for(ds):
    identity = auditor.collect_identity(ds, {})
    return identity, auditor.build_matchers(identity)


def test_identity_has_names_ids_dates_uids():
    identity, matchers = matchers_for(make_original())
    assert "JONES" in identity["name word"]
    assert "4107422692" in identity["id"]
    assert "12.11.1937" in identity["date"]
    assert "12 NOV 1937" in identity["date"]
    assert "1.2.3.4.5.6" in identity["uid"]
    assert "1.2.840.10008.5.1.4.1.1.2" not in identity["uid"]     # a class UID is not identifying


def test_finds_identity_in_other_forms():
    identity, matchers = matchers_for(make_original())
    kinds = [kind for kind, text in auditor.find_in_text("Erin Jones, born 11/12/1937", matchers)]
    assert "name word" in kinds
    assert "date" in kinds
    assert auditor.find_in_text("MRN 4107422692", matchers)[0][0] == "id"


def test_no_hit_inside_longer_words_or_numbers():
    identity, matchers = matchers_for(make_original())
    assert auditor.find_in_text("JONESBORO 141074226921", matchers) == []


def test_placeholders_are_not_identity():
    ds = make_original()
    ds.StationName = "NONE"
    identity, matchers = matchers_for(ds)
    assert "place or contact" not in identity


def test_k_anonymity_and_suppression():
    table = pd.DataFrame({"patient": ["a", "b", "c", "d"], "sex": ["F", "F", "F", "M"], "age": [40, 41, 42, 70],
                          "weight": [60.0, 61.0, 62.0, 90.0], "height": [1.6, 1.6, 1.6, 1.8],
                          "modalities": ["CT"] * 4, "manufacturers": ["GE"] * 4, "models": ["X"] * 4,
                          "body_parts": ["CHEST"] * 4, "n_studies": [1] * 4, "study_gaps": [""] * 4,
                          "study_year": [2010] * 4, "first_study_date": [None] * 4})
    assert reid_attack.singling_out(table, ["sex", "age"])["unique (k=1) %"] == 100.0
    anonymised, suppressed = reid_attack.anonymise(table, k=2)
    k = reid_attack.k_values(anonymised, reid_attack.ANONYMISED_QIS)
    assert k.min() >= 2 or (anonymised.loc[k < 2, "sex"] == "*").all()


def test_linking_with_ties():
    import numpy as np
    scores = np.array([3, 3, 1, 0])
    assert reid_attack.chance_in_top(scores, 0, 1) == 0.5      # two records tie for first place
    assert reid_attack.chance_in_top(scores, 2, 5) == 1.0
