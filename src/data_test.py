import sys
import pandas as pd
from evidently import Report
from evidently.presets.drift import DataDriftPreset
from evidently.presets.dataset_stats import DataSummaryPreset
import os

current = pd.read_csv("data/traffic.csv")
reference_path = "data/reference/traffic.csv"

if not os.path.exists(reference_path):
    print(f"Reference file not found. Copying from current data to {reference_path}.")
    os.makedirs(os.path.dirname(reference_path), exist_ok=True)
    current.to_csv(reference_path, index=False)

reference = pd.read_csv(reference_path)

del reference["Čas"]
del current["Čas"]

# Check if the reference and current data have the same columns
report = Report([
        DataSummaryPreset(),
        DataDriftPreset(),
    ],
    include_tests=True
)

# Run the report on the reference and current data
result = report.run(reference_data=reference, current_data=current)

# Save the report to an HTML file
result.save_html("src/reports/data_testing_report.html")

# Check if the report contains any tests and if all tests passed
all_tests_passed = True
result_dict = result.dict()
if "tests" in result_dict:
    for test in result_dict["tests"]:
        if "status" in test and test["status"] != "SUCCESS":
            all_tests_passed = False
            break

if not all_tests_passed:
    print("Data tests failed.")
    sys.exit(1)
else:
    print("Data tests passed.")
    # Replace the reference data with the current data
    os.remove(reference_path)
    current = pd.read_csv("data/traffic.csv")
    current.to_csv(reference_path, index=False)
sys.exit(0)