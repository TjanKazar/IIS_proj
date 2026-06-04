import csv
import re
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"

# Adjust only if your CSV schema changes
EXPECTED_COLUMNS = 9

# Values that often appear in the broken Lokacija field
STATUS_MARKERS = (
    "Ni prometa",
    "Normalen promet",
    "Gost promet",
    "Ni podatka",
)

DATE_TIME_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2}\b")
BIG_NUMBER_RE = re.compile(r"\d{4,}")


def load_params():
    if not PARAMS_PATH.exists():
        raise FileNotFoundError("params.yaml not found in project root")
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def is_header_row(row):
    lowered = {cell.strip().lower() for cell in row if cell.strip()}
    return "lokacija" in lowered or "status" in lowered or "timestamp" in lowered


def read_csv_with_optional_header(path):
    """
    Returns: (header_or_none, data_rows)
    """
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = [row for row in reader if any(cell.strip() for cell in row)]

    if not rows:
        return None, []

    if is_header_row(rows[0]):
        return rows[0], rows[1:]

    return None, rows


def repair_row_length(row):
    """
    Old scraper sometimes inserted extra commas into Lokacija.
    If there are too many columns, merge the overflow back into column 1.
    If there are too few, pad with empty strings.
    """
    row = [cell.strip() for cell in row]

    if len(row) > EXPECTED_COLUMNS:
        extra = len(row) - EXPECTED_COLUMNS
        # Merge surplus cells into Lokacija (index 1)
        merged_lokacija = "".join(row[1 : 1 + extra + 1]).strip()
        row = [row[0], merged_lokacija] + row[1 + extra + 1 :]

    if len(row) < EXPECTED_COLUMNS:
        row = row + [""] * (EXPECTED_COLUMNS - len(row))

    return row[:EXPECTED_COLUMNS]


def looks_corrupted(text):
    if not text:
        return False

    if DATE_TIME_RE.search(text):
        return True

    if any(marker in text for marker in STATUS_MARKERS):
        return True

    # Old scraper often glued a long numeric tail onto Lokacija
    if BIG_NUMBER_RE.search(text):
        return True

    return False


def clean_lokacija(row, previous_lokacija):
    """
    Rules:
    1. If Lokacija is empty, carry the value from the row above.
    2. If Lokacija looks corrupted and column 2 is contained inside it,
       use column 2 as the clean value.
    3. Otherwise strip obvious timestamp/status junk from Lokacija.
    """
    raw = row[1].strip() if len(row) > 1 else ""
    relacija = row[2].strip() if len(row) > 2 else ""

    if not raw:
        return previous_lokacija or ""

    if looks_corrupted(raw):
        if relacija and relacija in raw:
            return relacija

        cut_points = []
        m = DATE_TIME_RE.search(raw)
        if m:
            cut_points.append(m.start())

        for marker in STATUS_MARKERS:
            idx = raw.find(marker)
            if idx != -1:
                cut_points.append(idx)

        if cut_points:
            raw = raw[: min(cut_points)]

        raw = raw.rstrip(" ,;:-")

    return raw or previous_lokacija or ""


def normalize_row(row):
    """
    Strip whitespace everywhere so deduplication is stable.
    """
    return tuple(cell.strip() for cell in row)


def preprocess_traffic_data():
    params = load_params().get("preprocess", {})
    input_rel = params.get("input", "data/traffic.csv")
    output_rel = params.get("output", "data/preprocessed/traffic.csv")

    input_path = (ROOT_DIR / input_rel).resolve()
    output_path = (ROOT_DIR / output_rel).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    input_header, input_rows = read_csv_with_optional_header(input_path)
    if not input_rows:
        raise ValueError("Input CSV is empty")

    output_header = None
    existing_rows = []

    if output_path.exists():
        output_header, existing_rows = read_csv_with_optional_header(output_path)

    # Keep track of rows already written in the output dataset
    seen = set()
    cleaned_existing_rows = []

    previous_lokacija = ""

    for row in existing_rows:
        row = repair_row_length(row)
        row[1] = clean_lokacija(row, previous_lokacija)
        if row[1]:
            previous_lokacija = row[1]

        key = normalize_row(row)
        if key not in seen:
            seen.add(key)
            cleaned_existing_rows.append(row)

    cleaned_new_rows = []
    for row in input_rows:
        row = repair_row_length(row)
        row[1] = clean_lokacija(row, previous_lokacija)

        if row[1]:
            previous_lokacija = row[1]

        key = normalize_row(row)
        if key in seen:
            continue

        seen.add(key)
        cleaned_new_rows.append(row)

    final_rows = cleaned_existing_rows + cleaned_new_rows

    # Decide which header to write
    header_to_write = output_header or input_header

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if header_to_write:
            writer.writerow([cell.strip() for cell in header_to_write])
        writer.writerows(final_rows)

    print(
        f"Preprocess complete. "
        f"{len(cleaned_new_rows)} new rows added, "
        f"{len(final_rows)} total unique rows saved to {output_path}"
    )


if __name__ == "__main__":
    preprocess_traffic_data()