import csv
from pathlib import Path

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"


EXPECTED_COLUMNS = [
    "Cesta",
    "Lokacija",
    "Smer",
    "Št. vozil [N/h]",
    "Hitrost [km/h]",
    "Razmik [s]",
    "Zasedenost [%]",
    "Čas",
    "Stanje",
]


def load_params():
    if not PARAMS_PATH.exists():
        raise FileNotFoundError("params.yaml not found")

    with PARAMS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(value):
    if pd.isna(value):
        return value

    return " ".join(str(value).split())


def load_existing_keys(output_path):
    if not output_path.exists():
        return set()

    existing = set()

    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            existing.add(
                (
                    row["Lokacija"],
                    row["Smer"],
                    row["Čas"],
                )
            )

    return existing


def fetch_traffic_data():
    params = load_params()["fetch"]

    url = params["url"]

    output_path = (ROOT_DIR / params["output"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "TrafficCollector/1.0"
        },
    )
    response.raise_for_status()

    #
    # Repair malformed HTML first
    #
    soup = BeautifulSoup(
        response.text,
        "html5lib",
    )

    #
    # Let pandas parse the table
    #
    tables = pd.read_html(
        str(soup),
        flavor="bs4",
    )

    if not tables:
        raise RuntimeError("No tables found")

    df = tables[0].copy()

    #
    # Verify schema
    #
    if len(df.columns) != 9:
        raise RuntimeError(
            f"Expected 9 columns, got {len(df.columns)}"
        )

    df.columns = EXPECTED_COLUMNS

    #
    # Clean whitespace
    #
    for col in df.columns:
        df[col] = df[col].apply(normalize_text)

    #
    # Road column is omitted on continuation rows
    #
    df["Cesta"] = df["Cesta"].replace("", pd.NA)
    df["Cesta"] = df["Cesta"].ffill()

    # same for location

    df["Lokacija"] = df["Lokacija"].replace("", pd.NA)
    df["Lokacija"] = df["Lokacija"].ffill()

    #
    # Drop completely empty rows
    #
    df = df.dropna(
        how="all"
    )

    #
    # Remove accidental duplicate rows from page parsing
    #
    df = df.drop_duplicates()

    #
    # Existing deduplication
    #
    existing_keys = load_existing_keys(output_path)

    df["__key"] = list(
        zip(
            df["Lokacija"],
            df["Smer"],
            df["Čas"],
        )
    )

    new_df = df[
        ~df["__key"].isin(existing_keys)
    ].drop(columns="__key")

    #
    # Stable ordering
    #
    new_df = new_df.sort_values(
        ["Čas", "Cesta", "Lokacija", "Smer"]
    )

    file_exists = output_path.exists()

    if file_exists:
        new_df.to_csv(
            output_path,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8",
        )
    else:
        new_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )

    print(
        f"{len(new_df)} new rows added "
        f"({len(df) - len(new_df)} duplicates skipped)"
    )


if __name__ == "__main__":
    fetch_traffic_data()