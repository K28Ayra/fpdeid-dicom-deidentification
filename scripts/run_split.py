"""Run the frozen FP-DeID pipeline on one MIDI-B split (+ closed-loop auditor), then the official scorer.

Usage:  CUDA_VISIBLE_DEVICES=0 python scripts/run_split.py validation
        CUDA_VISIBLE_DEVICES=0 python scripts/run_split.py test
"""

import os
import sys

from fpdeid import paths
from fpdeid import pipeline
from fpdeid import scoring

split = sys.argv[1]
out_folder = os.path.join(paths.OUTPUTS, "phase5_" + split)

summary = pipeline.run_split(split, out_folder)
print(summary, flush=True)

# closed loop: the independent auditor checks every header and removes what it finds
audit_summary = pipeline.audit_and_fix(split, out_folder)
print(audit_summary, flush=True)

settings = pipeline.SPLITS[split]
db_path = scoring.run_official_scorer("phase5_" + split, out_folder, settings["answer_db"], settings["uid_map"],
                                      settings["patient_map"], os.path.join(paths.DERIVED, "scorer_runs"), cpus=16)
print("scorer results:", db_path, flush=True)
