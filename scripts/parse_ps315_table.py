"""Turn DICOM PS3.15 Table E.1-1 (from the official NEMA XML) into a CSV.

Usage:
    python scripts/parse_ps315_table.py tools/dicom_standard/part15.xml configs/ps315_table_e1_1.csv

Also writes configs/ps315_safe_private.csv (Table E.3.10-1, Safe Private Attributes).

The CSV has one row per attribute and one column per profile/option, e.g.
name, tag, basic_profile, retain_device_id, retain_long_modified_dates, clean_descriptors, ...
"""

import csv
import sys
import xml.etree.ElementTree as ET

NS = "{http://docbook.org/ns/docbook}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

COLUMNS = [
    "name",
    "tag",
    "retired",
    "in_std_iod",
    "basic_profile",
    "retain_safe_private",
    "retain_uids",
    "retain_device_id",
    "retain_institution_id",
    "retain_patient_chars",
    "retain_long_full_dates",
    "retain_long_modified_dates",
    "clean_descriptors",
    "clean_structured_content",
    "clean_graphics",
]


def cell_text(cell):
    # join all text inside the cell, and squeeze spaces
    pieces = []
    for text in cell.itertext():
        pieces.append(text)
    joined = " ".join(pieces)
    words = joined.split()
    return " ".join(words)


def main():
    xml_path = sys.argv[1]
    csv_path = sys.argv[2]

    tree = ET.parse(xml_path)
    root = tree.getroot()

    table = None
    for element in root.iter(NS + "table"):
        if element.get(XML_ID) == "table_E.1-1":
            table = element
            break

    body = table.find(NS + "tbody")
    rows = []
    for tr in body.findall(NS + "tr"):
        cells = tr.findall(NS + "td")
        values = [cell_text(c) for c in cells]
        if len(values) != len(COLUMNS):
            print("skipping row with", len(values), "cells:", values[:2])
            continue
        row = dict(zip(COLUMNS, values))
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("wrote", len(rows), "rows to", csv_path)



def parse_safe_private(xml_path, csv_path):
    """PS3.15 Table E.3.10-1 (Safe Private Attributes) -> CSV with group, element, creator, VR, meaning."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    table = None
    for element in root.iter(NS + "table"):
        if element.get(XML_ID) == "table_E.3.10-1":
            table = element
            break
    rows = []
    for tr in table.find(NS + "tbody").findall(NS + "tr"):
        values = [cell_text(c) for c in tr.findall(NS + "td")]
        if len(values) < 3:
            continue
        rows.append({"tag": values[0], "creator": values[1], "VR": values[2], "meaning": values[-1]})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "creator", "VR", "meaning"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("wrote", len(rows), "safe private rows to", csv_path)


if __name__ == "__main__":
    main()
    # the safe private table is written next to the main table
    safe_csv = sys.argv[2].replace("ps315_table_e1_1.csv", "ps315_safe_private.csv")
    parse_safe_private(sys.argv[1], safe_csv)
