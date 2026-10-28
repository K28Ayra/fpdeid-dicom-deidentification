"""Tests for the Phase 4 pixel decisions (no GPU and no OCR needed)."""

import numpy as np

from fpdeid import fingerprint, pixels


def make_fingerprint():
    collected = [("name", "PatientName", "JONES^ERIN"), ("date", "PatientBirthDate", "19371112"),
                 ("id", "PatientID", "4107422692")]
    return fingerprint.Fingerprint(collected, set())


def test_patient_name_and_birth_date_are_hidden():
    fp = make_fingerprint()
    assert pixels.decide("JONES ERIN [F] 01.12.2019", fp, "CR")[0] is True
    assert pixels.decide("DOB: 11.12.1937", fp, "CR")[0] is True


def test_ocr_mistakes_still_match():
    fp = make_fingerprint()
    hide, reason = pixels.decide("JONFS ERIM", fp, "CR")       # two letters misread
    assert hide is True
    assert "fuzzy" in reason


def test_markers_are_kept():
    fp = make_fingerprint()
    for text in ["L", "PORTABLE SEMI-ERECT", "AP", "5", "MLO"]:
        assert pixels.decide(text, fp, "DX")[0] is False


def test_initials_only_on_xray_and_not_common_words():
    assert pixels.decide("JKR", None, "DX")[0] is True
    assert pixels.decide("[mwf", None, "DX")[0] is True
    assert pixels.decide("JKR", None, "US")[0] is False          # ultrasound screen labels are not initials
    assert pixels.decide("SMM", None, "DX", common_codes={"SMM"})[0] is False


def test_label_words_and_dates():
    assert pixels.decide("ID: 12345", None, "US")[0] is True
    assert pixels.decide("Exam 03/04/2015", None, "US")[0] is True


def test_pieces_on_one_line_are_joined():
    found = [([10, 10, 40, 30], "ID:", 0.9), ([45, 11, 160, 31], "4107422692", 0.9), ([10, 200, 30, 220], "L", 0.9)]
    lines = pixels.group_lines(found)
    texts = sorted(line["text"] for line in lines)
    assert texts == ["ID: 4107422692", "L"]


def test_only_the_box_changes():
    image = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)
    before = image.copy()
    after = pixels.hide_boxes(image, [[20, 20, 40, 30]], "MONOCHROME2", margin=0)
    assert (after[20:30, 20:40] == before.min()).all()
    outside = np.ones((100, 100), dtype=bool)
    outside[20:30, 20:40] = False
    assert (after[outside] == before[outside]).all()


def test_ocr_noise_is_not_initials():
    assert pixels.decide("Lo C C", None, "DX", confidence=0.9)[0] is False      # several pieces, not one word
    assert pixels.decide("aud", None, "DX", confidence=0.2)[0] is False         # read with low confidence
    assert pixels.decide("JT", None, "DX", confidence=0.99)[0] is True
