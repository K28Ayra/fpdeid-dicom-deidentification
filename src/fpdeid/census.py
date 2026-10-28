"""Tag census: which tags exist in the data, how often, with which VR and example values."""

from multiprocessing import Pool

import pandas as pd
import pydicom

IGNORE_KEYWORDS = ["PixelData", "OverlayData", "FileMetaInformationVersion"]


def tag_label(element):
    """Like the scorer: (gggg,eeee) for standard tags, (gggg,"CREATOR",ee) for private tags."""
    group = element.tag.group
    elem = element.tag.element
    if element.tag.is_private and element.private_creator:
        creator = str(element.private_creator).upper()
        last_two = elem & 0xFF
        return "(%04X,\"%s\",%02X)" % (group, creator, last_two)
    return "(%04X,%04X)" % (group, elem)


def walk(ds, parent_path, rows):
    for element in ds:
        label = tag_label(element)
        if parent_path == "":
            path = label
        else:
            path = parent_path + ">" + label

        if element.keyword in IGNORE_KEYWORDS:
            value_text = "<binary>"
        elif element.VR == "SQ":
            value_text = "<sequence of %d>" % len(element.value)
        else:
            value_text = str(element.value).strip()

        row = {
            "path": path,
            "keyword": element.keyword,
            "name": element.name,
            "VR": element.VR,
            "private": element.tag.is_private,
            "depth": parent_path.count(">") + (0 if parent_path == "" else 1),
            "empty": value_text == "",
            "value": value_text[:80],
        }
        rows.append(row)

        if element.VR == "SQ":
            for item in element.value:
                walk(item, path, rows)


def census_one_file(file_path):
    rows = []
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        walk(ds, "", rows)
        modality = str(ds.get("Modality", ""))
    except Exception:
        modality = "unreadable"
    for row in rows:
        row["modality"] = modality
    # count each path once per file
    seen = {}
    for row in rows:
        if row["path"] not in seen:
            seen[row["path"]] = row
    return list(seen.values())


def build_census(file_paths, workers=16):
    """Returns one row per tag path with counts and up to 3 example values."""
    with Pool(workers) as pool:
        per_file = pool.map(census_one_file, file_paths, chunksize=50)

    counts = {}
    for file_rows in per_file:
        for row in file_rows:
            path = row["path"]
            if path not in counts:
                entry = {
                    "path": path,
                    "keyword": row["keyword"],
                    "name": row["name"],
                    "VR": row["VR"],
                    "private": row["private"],
                    "depth": row["depth"],
                    "n_files": 0,
                    "n_empty": 0,
                    "modalities": set(),
                    "examples": [],
                }
                counts[path] = entry
            entry = counts[path]
            entry["n_files"] = entry["n_files"] + 1
            if row["empty"]:
                entry["n_empty"] = entry["n_empty"] + 1
            entry["modalities"].add(row["modality"])
            new_example = not row["empty"] and row["value"] not in entry["examples"]
            if new_example and len(entry["examples"]) < 3:
                entry["examples"].append(row["value"])

    table = pd.DataFrame(list(counts.values()))
    table["modalities"] = table["modalities"].apply(lambda s: ",".join(sorted(s)))
    table["examples"] = table["examples"].apply(lambda e: " | ".join(e))
    table = table.sort_values("n_files", ascending=False).reset_index(drop=True)
    return table
