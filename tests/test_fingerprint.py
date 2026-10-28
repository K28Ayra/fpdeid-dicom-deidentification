"""Tests for the Phase 3 PHI fingerprint and its use inside the engine (no real data)."""

import json
import os

import pydicom

from fpdeid import engine, fingerprint, paths
from test_engine import make_file, make_policy


def make_fingerprint():
    collected = [
        ("name", "PatientName", "WILSON^AMY"),
        ("id", "PatientID", "8371727310"),
        ("id", "OtherPatientIDs", "689-96-0899"),
        ("institution", "InstitutionName", "Compton, Holmes and Jones Hospital"),
        ("address", "PatientAddress", "4643 Stern Dr Merced, CA 95348"),
        ("date", "StudyDate", "20100730"),
        ("name", "ReferringPhysicianName", "TORRES^MARK"),
    ]
    return fingerprint.Fingerprint(collected, common_words={"BRAIN"})


def test_name_is_removed_rest_is_kept():
    text, found = make_fingerprint().clean("PET-CT SERIES for Amy Wilson")
    assert text == "PET-CT SERIES"
    assert ("name", "PatientName") in found


def test_name_in_dicom_order_and_other_case():
    text, found = make_fingerprint().clean("AMY^WILSON")
    assert text == ""


def test_institution_acronym():
    text, found = make_fingerprint().clean("PET-CT STUDY at CHAJH")
    assert text == "PET-CT STUDY"
    assert ("institution", "InstitutionName") in found


def test_id_without_dashes_and_address():
    fp = make_fingerprint()
    assert fp.clean("ABDOMEN for 689960899")[0] == "ABDOMEN"
    assert fp.clean("LUNG STAGING : Amy Wilson : 4643 Stern Dr Merced, CA 95348")[0] == "LUNG STAGING"


def test_date_in_other_format():
    fp = make_fingerprint()
    assert fp.clean("Performed 07/30/2010")[0] == "Performed"
    assert fp.clean("CT COLONOGRAPHY SCREEN 20100730")[0] == "CT COLONOGRAPHY SCREEN"


def test_doctor_and_phone_patterns():
    fp = make_fingerprint()
    assert fp.clean("Ordered for 689-96-0899 by Dr. Torres")[0] == "Ordered for"
    assert fp.clean("Followup for Dr. Ryan, call 1-445-271-7733")[0] == "Followup for, call"


def test_medical_words_are_not_removed():
    fp = make_fingerprint()
    assert fp.clean("MR BRAIN W/O CONTRAST")[0] == "MR BRAIN W/O CONTRAST"
    assert fp.clean("AXIAL T2 FLAIR")[1] == []


def test_acronym():
    assert fingerprint.acronym("Compton, Holmes and Jones Hospital") == "CHAJH"
    assert fingerprint.acronym("Malone-Smith Community Hospital") == "MSCH"


def test_engine_cleans_text_with_fingerprint(tmp_path):
    folder = str(tmp_path)
    input_path = make_file(folder)
    policy = make_policy(folder)
    policy = engine.Policy(os.path.join(paths.CONFIGS, "policy.yaml"), os.path.join(folder, "uids.csv"),
                           os.path.join(folder, "patients.csv"))
    fingerprints, common = fingerprint.build_all([input_path], workers=1)
    policy.set_fingerprints(fingerprints)

    out_root = os.path.join(folder, "out")
    table = engine.run([input_path], out_root, policy, workers=1)
    assert table["error"].iloc[0] == ""
    ds = pydicom.dcmread(table["output"].iloc[0])

    # the description is kept, only the name is gone
    assert ds.SeriesDescription == "CT CHEST"
    # private text: kept, but without the name
    block = ds.private_block(0x0009, "TEST CREATOR")
    assert block[0x02].value == "home address"

    with open(os.path.join(out_root, "audit.jsonl")) as f:
        record = json.loads(f.readline())
    cleaned = [c for c in record["changes"] if c["action"] == "F"]
    assert len(cleaned) >= 2
    for change in cleaned:
        assert "PatientName" in change["reason"]       # every removal says which fingerprint entry it matched


def test_more_patterns():
    fp = make_fingerprint()
    assert fp.clean("Followup, call (606)066-5106x76")[0] == "Followup, call"
    assert fp.clean("1ACRIN_6664_DR_ZIMMERMAN")[0] == "1ACRIN_6664"
    assert fp.clean("Ordered for 3363075481 by AH")[0] == "Ordered for"     # ID and staff initials removed


def test_site_level_names_of_other_patients():
    collected = {
        "p1": [("name", "PatientName", "WILSON^AMY")],
        "p2": [("name", "PatientName", "WILLIAMS^MARGARET"), ("address", "PatientAddress", "338 Rosston Rd Prescott, AR 71857")],
    }
    site = fingerprint.site_level_entries(collected)
    fp = fingerprint.Fingerprint(collected["p1"], set(), site_entries=site)
    text, found = fp.clean("NONIONIC : Margaret Williams : 338 Rosston Rd Prescott, AR 71857")
    assert text == "NONIONIC"
    # single words of other patients are NOT used (they could be ordinary words)
    assert fp.clean("WILLIAMS protocol")[0] == "WILLIAMS protocol"
