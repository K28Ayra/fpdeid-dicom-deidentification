"""Download one TCIA series image by image (for very large series whose zip download gets cut off).

Usage:
    python scripts/download_series_by_image.py SERIES_UID PATIENT_ID OUT_DIR
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def download_image(job):
    series_uid, sop_uid, folder = job
    path = os.path.join(folder, sop_uid + ".dcm")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skipped"
    params = {"SeriesInstanceUID": series_uid, "SOPInstanceUID": sop_uid}
    for attempt in range(5):
        try:
            response = requests.get(API + "/getSingleImage", params=params, timeout=300)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            return "ok"
        except Exception:
            pass
    return "FAILED"


def main():
    series_uid, patient_id, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    folder = os.path.join(out_dir, patient_id, series_uid)
    os.makedirs(folder, exist_ok=True)
    response = requests.get(API + "/getSOPInstanceUIDs", params={"SeriesInstanceUID": series_uid}, timeout=120)
    sop_uids = [row["SOPInstanceUID"] for row in response.json()]
    print("images in series:", len(sop_uids), flush=True)
    jobs = [(series_uid, sop, folder) for sop in sop_uids]
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(download_image, jobs))
    print({status: results.count(status) for status in set(results)})


if __name__ == "__main__":
    main()
