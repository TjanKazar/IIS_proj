import csv
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_params():
    if not PARAMS_PATH.exists():
        raise FileNotFoundError("params.yaml not found in project root")
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def fetch_traffic_data():
    params = load_params().get("fetch", {})
    url = params.get("url")
    if not url:
        raise ValueError("Missing fetch.url in params.yaml")

    output_rel = params.get("output", "data/traffic.csv")
    output_path = (ROOT_DIR / output_rel).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("No table found in the response")

    all_rows = [
        [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        for tr in table.find_all("tr")
        if tr.find_all(["td", "th"])
    ]
    if not all_rows:
        raise RuntimeError("No rows found in the table")

    header = all_rows[0]
    data_rows = all_rows[1:]

    last_road = ""
    for row in data_rows:
        if row and row[0]:
            last_road = row[0]
        elif row:
            row[0] = last_road

    # Columns: road(0), location(1), direction(2), ..., time(7)
    # Dedup key: location + direction + time
    def row_key(row):
        if len(row) < 8:
            return tuple(row)
        return (row[1], row[2], row[7])

    existing_keys = set()
    file_exists = output_path.exists()
    if file_exists:
        with output_path.open(newline="", encoding="utf-8") as file:
            for row in csv.reader(file):
                if len(row) < 8:
                    existing_keys.add(tuple(row))
                else:
                    existing_keys.add((row[1], row[2], row[7]))

    new_rows = [r for r in data_rows if row_key(r) not in existing_keys]

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(new_rows)

    print(
        f"{len(new_rows)} new rows added ({len(data_rows) - len(new_rows)} duplicates skipped)"
    )


if __name__ == "__main__":
    fetch_traffic_data()
 
