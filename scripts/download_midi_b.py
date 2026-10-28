"""Download a MIDI-B collection from the public TCIA (NBIA) REST API.

Usage:
    python scripts/download_midi_b.py MIDI-B-Synthetic-Validation data/midi_b/validation/images
"""

import io
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def get_series_list(collection):
    url = API + "/getSeries"
    params = {"Collection": collection}
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def download_one_series(series, out_dir):
    series_uid = series["SeriesInstanceUID"]
    patient_id = series["PatientID"]
    expected = int(series["ImageCount"])

    folder = os.path.join(out_dir, patient_id, series_uid)
    if os.path.isdir(folder):
        already = len(os.listdir(folder))
        if already >= expected:
            return series_uid, "skipped", already

    url = API + "/getImage"
    params = {"SeriesInstanceUID": series_uid}

    # try 3 times, the server sometimes drops connections
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=600)
            response.raise_for_status()
            zip_bytes = io.BytesIO(response.content)
            os.makedirs(folder, exist_ok=True)
            with zipfile.ZipFile(zip_bytes) as z:
                z.extractall(folder)
            count = len(os.listdir(folder))
            return series_uid, "ok", count
        except Exception as error:
            last_error = str(error)

    return series_uid, "FAILED: " + last_error, 0


def main():
    collection = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    series_list = get_series_list(collection)
    print(collection, ":", len(series_list), "series", flush=True)

    # keep the series list next to the images (useful for the inventory)
    list_path = os.path.join(out_dir, "..", "series_list.json")
    with open(list_path, "w") as f:
        json.dump(series_list, f, indent=1)

    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        jobs = [pool.submit(download_one_series, s, out_dir) for s in series_list]
        for job in jobs:
            series_uid, status, count = job.result()
            done = done + 1
            if status.startswith("FAILED"):
                failed.append(series_uid)
            print(done, "/", len(series_list), series_uid, status, count, flush=True)

    print("finished. failed series:", len(failed))
    for uid in failed:
        print("  ", uid)


if __name__ == "__main__":
    main()
