import csv
import re
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"


DATE_TIME_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2}\b")


def load_params():
    if not PARAMS_PATH.exists():
        raise FileNotFoundError("params.yaml not found")

    with PARAMS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(cell.strip() for cell in r)]

    if not rows:
        return None, []

    header = rows[0]
    data = rows[1:] if "Lokacija" in header else rows
    return header if "Lokacija" in header else None, data


def normalize_row_length(row, expected=9):
    row = [c.strip() for c in row]

    if len(row) < expected:
        row += [""] * (expected - len(row))

    return row[:expected]


def forward_fill_columns(rows):
    """
    Fix missing Cesta, Lokacija, Čas by propagating previous values.
    """
    prev_cesta = ""
    prev_lokacija = ""
    prev_cas = ""

    for row in rows:
        # ensure correct length
        row = normalize_row_length(row, 9)

        cesta, lokacija, smer, vozila, hitrost, razmik, zasedenost, cas, stanje = row

        if cesta:
            prev_cesta = cesta
        else:
            row[0] = prev_cesta

        if lokacija:
            prev_lokacija = lokacija
        else:
            row[1] = prev_lokacija

        # Čas sometimes missing only visually (same block)
        if cas and DATE_TIME_RE.search(cas):
            prev_cas = cas
        else:
            row[7] = prev_cas

        yield row


def preprocess():
    params = load_params().get("preprocess", {})

    input_path = (ROOT_DIR / params.get("input", "data/raw.csv")).resolve()
    output_path = (ROOT_DIR / params.get("output", "data/preprocessed.csv")).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    header, rows = read_csv(input_path)

    if not rows:
        raise ValueError("Input CSV is empty")

    cleaned = list(forward_fill_columns(rows))

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if header:
            writer.writerow(header)

        writer.writerows(cleaned)

    print(f"Preprocessed {len(cleaned)} rows → {output_path}")


if __name__ == "__main__":
    preprocess()