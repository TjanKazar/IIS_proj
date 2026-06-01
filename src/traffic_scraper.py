import csv
import json
import sys
from datetime import timezone
import requests
from bs4 import BeautifulSoup
import os
from pathlib import Path

URL = "http://portal.drsc.si/traffic/loclist_si.htm"

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR.parent / "data" / "traffic.csv"

response = requests.get(URL)
soup = BeautifulSoup(response.text, "html.parser")
table = soup.find("table")
 
all_rows = [
    [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
    for tr in table.find_all("tr")
    if tr.find_all(["td", "th"])
]
 
header = all_rows[0]
data_rows = all_rows[1:]
 
# Columns: road(0), location(1), direction(2), ..., time(7)
# Dedup key: location + direction + time
def row_key(row):
    return (row[1], row[2], row[7])
 
# Load existing keys from CSV if it exists
existing_keys = set()
file_exists = os.path.exists(CSV_FILE)
 
if file_exists:
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            existing_keys.add((row[1], row[2], row[7]))
 
new_rows = [r for r in data_rows if row_key(r) not in existing_keys]
 
with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(header)
    writer.writerows(new_rows)
 
print(f"{len(new_rows)} new rows added ({len(data_rows) - len(new_rows)} duplicates skipped)")
