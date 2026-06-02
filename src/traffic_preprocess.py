import csv
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_params():
    if not PARAMS_PATH.exists():
        raise FileNotFoundError("params.yaml not found in project root")
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def preprocess_traffic_data():
    params = load_params().get("preprocess", {})
    input_rel = params.get("input", "data/traffic.csv")
    output_rel = params.get("output", "data/preprocessed/traffic.csv")

    input_path = (ROOT_DIR / input_rel).resolve()
    output_path = (ROOT_DIR / output_rel).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV is empty")

    header = rows[0]
    data_rows = [row for row in rows[1:] if any(cell.strip() for cell in row)]

    deduped_rows = []
    seen = set()
    for row in data_rows:
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        deduped_rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(deduped_rows)

    print(
        f"Preprocess complete. {len(deduped_rows)} rows saved to {output_path}"
    )


if __name__ == "__main__":
    preprocess_traffic_data()
