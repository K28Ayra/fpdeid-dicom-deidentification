"""FP-DeID in one command (Phase 6).

    python -m fpdeid.run --input DIR --output DIR [--policy configs/policy.yaml]
                         [--uid-map CSV] [--patient-map CSV] [--workers 16] [--no-pixels] [--no-auditor]

Steps (each one is logged, and the run can be stopped and started again):
  1. fingerprints : read every input header once and build each patient's PHI fingerprint
  2. headers      : clean every file (rules + fingerprint). Files already in audit.jsonl are skipped (resume).
  3. pixels       : OCR on images that may have burned-in text; hide only identifying lines (needs a GPU to be fast).
                    OCR results are saved in ocr_cache.jsonl, so a stopped run does not read the same image twice.
  4. auditor      : the independent auditor checks every output header and removes what it finds.
  5. lineage      : lineage.jsonl, one line per file: input SHA-256 -> output SHA-256, policy hash, code version, time,
                    and where its audit entries are.

Files written in --output:
  <new patient>/<new series>/<new SOP UID>.dcm, audit.jsonl, audit_pixels.jsonl, audit_auditor_fixes.jsonl,
  ocr_cache.jsonl, lineage.jsonl, run_summary.json, run.log

The pseudonym key (the mapping tables, or the secret in FPDEID_SECRET) stays with the hospital; it is never written
to --output.
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from multiprocessing import Pool

import psutil
import pydicom

from fpdeid import engine
from fpdeid import fingerprint
from fpdeid import paths

LOG = logging.getLogger("fpdeid")


# ---------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------

def setup_logging(output_folder):
    LOG.setLevel(logging.INFO)
    LOG.handlers = []
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    to_file = logging.FileHandler(os.path.join(output_folder, "run.log"))
    to_file.setFormatter(formatter)
    to_console = logging.StreamHandler()
    to_console.setFormatter(formatter)
    LOG.addHandler(to_file)
    LOG.addHandler(to_console)


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def code_version():
    """The git commit of the code (plus '-dirty' if files were changed after it), or 'unknown'."""
    try:
        result = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=paths.PROJECT,
                                capture_output=True, text=True, timeout=10)
        version = result.stdout.strip()
        if result.returncode == 0 and version != "":
            return version
    except Exception:
        pass
    return "unknown"


def is_dicom(path):
    """True if the file has the DICOM preamble ('DICM' at byte 128)."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def find_dicom_files(input_folder):
    input_folder = os.path.abspath(input_folder)
    found = []
    for folder, subfolders, names in os.walk(input_folder):
        subfolders.sort()
        for name in sorted(names):
            path = os.path.join(folder, name)
            if is_dicom(path):
                found.append(path)
    return found


class MemoryWatch:
    """Peak memory of this process and all its worker processes, measured every second.

    PSS (proportional set size) is used: memory that several processes share is counted once, split between them.
    (Adding up RSS would count shared memory once per process and overstate the total.)
    """

    def __init__(self):
        self.peak_bytes = 0
        self.running = True
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def current(self):
        main = psutil.Process()
        total = 0
        for process in [main] + main.children(recursive=True):
            try:
                total = total + process.memory_full_info().pss
            except psutil.Error:
                pass
        return total

    def watch(self):
        while self.running:
            self.peak_bytes = max(self.peak_bytes, self.current())
            time.sleep(1)

    def stop(self):
        self.running = False
        self.thread.join()
        return round(self.peak_bytes / 1e9, 2)


def audit_records(output_folder):
    """(line number, input, output, error) for every complete line of audit.jsonl, read one line at a time.

    The audit log can be hundreds of MB (every change of every file), so it is never loaded as a whole.
    """
    path = os.path.join(output_folder, "audit.jsonl")
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for number, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            records.append((number, record["input"], record["output"], record["error"]))
    return records


def read_jsonl(path):
    """All complete lines of a JSON-lines file (a half-written last line of a stopped run is ignored)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


# ---------------------------------------------------------------
# Steps
# ---------------------------------------------------------------

def patient_of_files(input_files, workers):
    with Pool(workers) as pool:
        ids = pool.map(read_patient_id, input_files, chunksize=50)
    return dict(zip(input_files, ids))


def read_patient_id(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["PatientID"])
    return str(ds.get("PatientID", ""))


def clean_headers(input_files, output_folder, policy, workers):
    """Step 2. Appends one line per file to audit.jsonl; files already there are not cleaned again."""
    audit_path = os.path.join(output_folder, "audit.jsonl")
    done = set()
    for number, input_path, output_path, error in audit_records(output_folder):
        if error == "" and os.path.exists(output_path):
            done.add(input_path)
    # rewrite the log without broken lines or lines of files that will be cleaned again (line by line)
    if os.path.exists(audit_path):
        temporary = audit_path + ".tmp"
        with open(audit_path) as source, open(temporary, "w") as target:
            for line in source:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record["input"] in done:
                    target.write(line if line.endswith("\n") else line + "\n")
        os.replace(temporary, audit_path)

    to_do = [path for path in input_files if path not in done]
    LOG.info("headers: %d files already done, %d to clean", len(done), len(to_do))
    errors = 0
    with open(audit_path, "a") as audit_file:
        with Pool(workers, initializer=engine.start_worker, initargs=(policy, output_folder)) as pool:
            for count, record in enumerate(pool.imap(engine.clean_file, to_do, chunksize=20), start=1):
                audit_file.write(json.dumps(record) + "\n")
                audit_file.flush()
                if record["error"] != "":
                    errors = errors + 1
                    LOG.warning("could not clean %s: %s", record["input"], record["error"])
                if count % 5000 == 0:
                    LOG.info("headers: %d / %d", count, len(to_do))
    return errors


def clean_pixels(output_folder, fingerprints, patient_of, workers):
    """Step 3: burned-in text."""
    from fpdeid import pixels

    records = []
    for number, input_path, output_path, error in audit_records(output_folder):
        if output_path != "":
            records.append({"input": input_path, "output": output_path})
    modality_of = {}
    series_of = {}
    for record in records:
        ds = pydicom.dcmread(record["input"], stop_before_pixels=True, specific_tags=["Modality", "SeriesInstanceUID"])
        modality_of[record["input"]] = str(ds.get("Modality", ""))
        series_of[record["input"]] = str(ds.get("SeriesInstanceUID", ""))
    pixel_audit = os.path.join(output_folder, "audit_pixels.jsonl")
    if os.path.exists(pixel_audit):
        os.remove(pixel_audit)          # written again from the saved OCR results
    summary = pixels.clean_outputs(records, fingerprints, patient_of, modality_of, series_of, pixel_audit,
                                   method="fpdeid", ocr_cache_path=os.path.join(output_folder, "ocr_cache.jsonl"))
    return summary


def run_auditor(output_folder, patient_of, workers):
    """Step 4: the independent auditor, and the closed-loop fix."""
    from fpdeid import auditor

    originals = {}
    outputs = {}
    for number, input_path, output_path, error in audit_records(output_folder):
        if output_path == "":
            continue
        patient = patient_of[input_path]
        originals.setdefault(patient, []).append(input_path)
        outputs.setdefault(patient, []).append(output_path)
    jobs = [(originals[p], outputs[p]) for p in sorted(originals)]
    hits, elements = auditor.audit_headers(jobs, workers=workers)
    fix_log = os.path.join(output_folder, "audit_auditor_fixes.jsonl")
    open(fix_log, "w").close()
    fixed = auditor.fix_header_hits(hits, fix_log)
    hits_after, elements_after = auditor.audit_headers(jobs, workers=workers)
    return {"auditor elements read": elements, "auditor hits": len(hits), "auditor fixed files": fixed,
            "auditor hits after fix": len(hits_after)}


def lineage_line(job):
    input_path, output_path, details = job
    entry = dict(details)
    entry["input"] = input_path
    entry["input_sha256"] = sha256_of_file(input_path)
    entry["output"] = output_path
    entry["output_sha256"] = sha256_of_file(output_path)
    return entry


def write_lineage(output_folder, policy_path, version, started, workers):
    """Step 5: one line per output file."""
    policy_hash = sha256_of_file(policy_path)
    pixel_line = {}
    for number, entry in enumerate(read_jsonl(os.path.join(output_folder, "audit_pixels.jsonl")), start=1):
        pixel_line[entry["output"]] = number
    fix_lines = {}
    for number, entry in enumerate(read_jsonl(os.path.join(output_folder, "audit_auditor_fixes.jsonl")), start=1):
        fix_lines.setdefault(entry["file"], []).append(number)

    jobs = []
    for number, input_path, output_path, error in audit_records(output_folder):
        if output_path == "":
            continue
        details = {
            "policy": os.path.basename(policy_path),
            "policy_sha256": policy_hash,
            "code_version": version,
            "run_started": started,
            "audit_line": number,
            "audit_pixels_line": pixel_line.get(output_path),
            "auditor_fix_lines": fix_lines.get(output_path, []),
        }
        jobs.append((input_path, output_path, details))
    with Pool(workers) as pool:
        lines = pool.map(lineage_line, jobs, chunksize=50)
    with open(os.path.join(output_folder, "lineage.jsonl"), "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return len(lines)


# ---------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------

def run(input_folder, output_folder, policy_path, uid_map=None, patient_map=None, workers=16, do_pixels=True,
        do_auditor=True):
    os.makedirs(output_folder, exist_ok=True)
    setup_logging(output_folder)
    memory = MemoryWatch()
    started = datetime.datetime.now().isoformat(timespec="seconds")
    version = code_version()
    summary = {"input": input_folder, "output": output_folder, "policy": policy_path, "code_version": version,
               "started": started, "workers": workers}
    LOG.info("FP-DeID run: %s -> %s (policy %s, code %s, %d workers)", input_folder, output_folder,
             os.path.basename(policy_path), version, workers)

    start = time.time()
    input_files = find_dicom_files(input_folder)
    patient_of = patient_of_files(input_files, workers)
    summary["files"] = len(input_files)
    summary["patients"] = len(set(patient_of.values()))
    LOG.info("found %d DICOM files of %d patients", summary["files"], summary["patients"])

    policy = engine.Policy(policy_path, uid_map, patient_map)
    fingerprints = {}
    if policy.use_fingerprint:
        fingerprints, common_words = fingerprint.build_all(input_files, workers=workers,
                                                           common_share=policy.common_word_share,
                                                           use_site=policy.site_level)
        policy.set_fingerprints(fingerprints)
    summary["seconds: fingerprints"] = round(time.time() - start, 1)
    LOG.info("fingerprints built for %d patients", len(fingerprints))

    start = time.time()
    summary["header errors"] = clean_headers(input_files, output_folder, policy, workers)
    summary["seconds: headers"] = round(time.time() - start, 1)
    LOG.info("headers done in %.1f s (%d errors)", summary["seconds: headers"], summary["header errors"])

    if do_pixels:
        start = time.time()
        pixel_summary = clean_pixels(output_folder, fingerprints, patient_of, workers)
        summary.update(pixel_summary)
        summary["seconds: pixels"] = round(time.time() - start, 1)
        LOG.info("pixels done in %.1f s: %s", summary["seconds: pixels"], pixel_summary)

    if do_auditor:
        start = time.time()
        auditor_summary = run_auditor(output_folder, patient_of, workers)
        summary.update(auditor_summary)
        summary["seconds: auditor"] = round(time.time() - start, 1)
        LOG.info("auditor done in %.1f s: %s", summary["seconds: auditor"], auditor_summary)

    start = time.time()
    summary["lineage lines"] = write_lineage(output_folder, policy_path, version, started, workers)
    summary["seconds: lineage"] = round(time.time() - start, 1)

    summary["peak memory GB (all processes)"] = memory.stop()
    with open(os.path.join(output_folder, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    LOG.info("run finished: %s", json.dumps(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description="FP-DeID: fingerprint-guided, closed-loop DICOM de-identification")
    parser.add_argument("--input", required=True, help="folder with the original DICOM files (searched recursively)")
    parser.add_argument("--output", required=True, help="folder for the de-identified files and the logs")
    parser.add_argument("--policy", default=os.path.join(paths.CONFIGS, "policy.yaml"))
    parser.add_argument("--uid-map", default=None, help="CSV id_old,id_new with the UID pseudonyms (optional)")
    parser.add_argument("--patient-map", default=None, help="CSV id_old,id_new with the patient pseudonyms (optional)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-pixels", action="store_true", help="skip the burned-in text step (no GPU needed)")
    parser.add_argument("--no-auditor", action="store_true", help="skip the independent auditor")
    args = parser.parse_args()
    run(args.input, args.output, args.policy, args.uid_map, args.patient_map, args.workers,
        do_pixels=not args.no_pixels, do_auditor=not args.no_auditor)


if __name__ == "__main__":
    main()
