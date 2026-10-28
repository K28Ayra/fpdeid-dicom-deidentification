"""DICOM validity check with dciodvfy (dicom3tools).

We count "Error" lines for the input file and for the output file.
A file is valid for us if the output has NO NEW error compared with its input
(some synthetic input files already have errors, we should not be blamed for them).
"""

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fpdeid import paths

DCIODVFY = os.path.join(paths.PROJECT, "tools", "dicom3tools_deb", "root", "usr", "bin", "dciodvfy")


def error_lines(file_path):
    """Return the list of error messages dciodvfy prints for one file."""
    result = subprocess.run([DCIODVFY, file_path], capture_output=True, text=True)
    output = result.stdout + result.stderr
    errors = []
    for line in output.splitlines():
        if line.startswith("Error"):
            # drop the value part (" = <...>") so input and output errors can be compared
            signature = re.sub(r" = <[^>]*>", "", line.strip())
            # UIDs also appear inside some messages; they change on purpose, so mask them
            signature = re.sub(r"[0-9]+(\.[0-9]+){3,}", "<uid>", signature)
            errors.append(signature)
    return errors


def compare_pair(pair):
    input_path, output_path = pair
    input_errors = error_lines(input_path)
    output_errors = error_lines(output_path)
    new_errors = []
    for line in output_errors:
        if line not in input_errors:
            new_errors.append(line)
    row = {
        "input": input_path,
        "output": output_path,
        "input_errors": len(input_errors),
        "output_errors": len(output_errors),
        "new_errors": len(new_errors),
        "new_error_examples": " || ".join(new_errors[:3]),
    }
    return row


def check_pairs(pairs, workers=16):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(compare_pair, pairs))
    return pd.DataFrame(rows)
