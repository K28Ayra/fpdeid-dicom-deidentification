"""All important folders in one place, so notebooks and scripts agree."""

import os

# the project folder is found from this file (src/fpdeid/paths.py), so the code runs from any location
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# data and large outputs can live elsewhere (e.g. a bigger disk): export FPDEID_DATA=/path/to/data
DATA = os.environ.get("FPDEID_DATA", os.path.join(PROJECT, "ayra_data"))
MIDI_B = os.path.join(DATA, "midi_b")

VALIDATION_IMAGES = os.path.join(MIDI_B, "validation", "images")
VALIDATION_ANSWER_DIR = os.path.join(MIDI_B, "validation", "answer_key")
VALIDATION_SERIES_LIST = os.path.join(MIDI_B, "validation", "series_list.json")
VALIDATION_PATIENT_MAP = os.path.join(MIDI_B, "MIDI-B-Patient-Mapping-Validation.csv")
VALIDATION_UID_MAP = os.path.join(MIDI_B, "MIDI-B-UID-Mapping-Validation.csv")

DERIVED = os.path.join(DATA, "derived")      # big tables we make (not in git)
OUTPUTS = os.path.join(DATA, "outputs")      # de-identified files (not in git)
RESULTS = os.path.join(PROJECT, "results")   # small tables + plots (in git)
CONFIGS = os.path.join(PROJECT, "configs")

SCORER_DIR = os.path.join(PROJECT, "tools", "MIDI_validation_script")
# python of the environment with the official scorer (conda env "midi-scorer", see README)
SCORER_PYTHON = os.environ.get("SCORER_PYTHON", os.path.join(PROJECT, "tools", "scorer_venv", "bin", "python"))

TEST_IMAGES = os.path.join(MIDI_B, "test", "images")
TEST_ANSWER_DIR = os.path.join(MIDI_B, "test", "answer_key")
TEST_PATIENT_MAP = os.path.join(MIDI_B, "MIDI-B-Patient-Mapping-Test.csv")
TEST_UID_MAP = os.path.join(MIDI_B, "MIDI-B-UID-Mapping-Test.csv")
