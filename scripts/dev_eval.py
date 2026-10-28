"""Quick loop for development: clean dev-mini with the engine, score it, print the main failures.

Usage:
    python scripts/dev_eval.py RUN_NAME [policy.yaml]
"""

import os
import pickle
import shutil
import sys
import time

import pandas as pd

from fpdeid import engine, fingerprint, paths, scoring


def main():
    run_name = sys.argv[1]
    policy_path = os.path.join(paths.CONFIGS, "policy.yaml")
    if len(sys.argv) > 2:
        policy_path = sys.argv[2]

    file_table = pd.read_csv(os.path.join(paths.DERIVED, "validation_file_table.csv"), dtype=str)
    with open(os.path.join(paths.DERIVED, "dev_mini_patients.txt")) as f:
        dev_patients = f.read().split()
    dev_files = file_table[file_table["patient_id"].isin(dev_patients)]

    policy = engine.Policy(policy_path, paths.VALIDATION_UID_MAP, paths.VALIDATION_PATIENT_MAP)
    if policy.use_fingerprint:
        # fingerprints are built once from ALL validation files (so word statistics use every patient)
        cache = os.path.join(paths.DERIVED, "fingerprints_validation.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                fingerprints, common_words = pickle.load(f)
        else:
            start = time.time()
            fingerprints, common_words = fingerprint.build_all(file_table["path"].tolist(), workers=16,
                                                              common_share=policy.common_word_share,
                                                              use_site=policy.site_level)
            with open(cache, "wb") as f:
                pickle.dump((fingerprints, common_words), f)
            print("fingerprints built in", round(time.time() - start, 1), "s;", len(common_words), "common words")
        policy.set_fingerprints(fingerprints)

    out_folder = os.path.join(paths.OUTPUTS, run_name)
    shutil.rmtree(out_folder, ignore_errors=True)

    start = time.time()
    run_table = engine.run(dev_files["path"].tolist(), out_folder, policy, workers=16)
    print("cleaned", len(run_table), "files in", round(time.time() - start, 1), "s, errors:",
          (run_table["error"] != "").sum())

    start = time.time()
    db_path = scoring.run_official_scorer(run_name, out_folder, os.path.join(paths.DERIVED, "answer_dev_mini.db"),
                                          paths.VALIDATION_UID_MAP, paths.VALIDATION_PATIENT_MAP,
                                          os.path.join(paths.DERIVED, "scorer_runs"), cpus=16)
    print("scored in", round(time.time() - start), "s")

    results = scoring.add_categories(scoring.load_results(db_path))
    summary = scoring.summarise(results)
    summary.update(scoring.privacy_utility(results))
    print(pd.Series(summary).to_string())

    failed = results[results["check_passed"].fillna(0).astype(float) < 1]
    for category in ["hipaa", "dicom", "tcia"]:
        part = failed[failed["score_cat"] == category]
        print("\n== failures in", category, ":", len(part))
        top = part.groupby(["action", "tag_name"]).size().sort_values(ascending=False).head(12)
        print(top.to_string())

    results.to_pickle(os.path.join(paths.DERIVED, run_name + "_results.pkl"))


if __name__ == "__main__":
    main()
