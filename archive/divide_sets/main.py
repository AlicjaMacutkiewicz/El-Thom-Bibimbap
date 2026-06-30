import os
import random
import re
import shutil
from collections import defaultdict

LOG_FILE = "generated_flights.txt"
SOURCE_DIR = "source_data"
FILE_PREFIX = "flight_"
FILE_EXT = ".parquet"
TESTING_DIR = "testing_data"
TRAINING_DIR = "training_data"


def process_logs():
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] could not find {LOG_FILE}, please check the path")
        return

    with open(LOG_FILE) as f:
        log_content = f.read()

    DATE_PATTERN = r"^(\d{4}-\d{2}-\d{2})$"
    dates = re.findall(DATE_PATTERN, log_content, re.MULTILINE)

    unique_dates = list(set(dates))

    if not unique_dates:
        print("no valid dates found in the log file.")
        return

    year_counts = defaultdict(int)
    for date in unique_dates:
        year = date.split("-")[0]
        year_counts[year] += 1

    print("--- year counts ---")
    for year, count in sorted(year_counts.items()):
        if count == 24:
            print(f"[OK] year {year} has exactly 24 entries.")
        else:
            print(f"[WARNING] year {year} has {count} entries (Expected 24).")

    valid_dates = []
    missing_files = []
    print("\n--- file check ---")
    for date in unique_dates:
        filename = f"{FILE_PREFIX}{date}{FILE_EXT}"
        file_path = os.path.join(SOURCE_DIR, filename)
        if os.path.exists(file_path):
            valid_dates.append(date)
        else:
            missing_files.append(filename)

    if missing_files:
        print(f"[WARNING] {len(missing_files)} files were missing in the source directory.")
        for mf in missing_files[:5]:
            print(f"  missing: {mf}")
        if len(missing_files) > 5:
            print("  ...")

    if not valid_dates:
        print("no matching files found to process")
        return

    print(f"\nfound {len(valid_dates)} matching files out of {len(unique_dates)} dates logged")

    os.makedirs(TESTING_DIR, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)

    valid_dates.sort()
    random.shuffle(valid_dates)

    split_index = int(len(valid_dates) * 0.2)
    testing_dates = valid_dates[:split_index]
    training_dates = valid_dates[split_index:]

    print("\n--- distributing files ---")

    for date in testing_dates:
        filename = f"{FILE_PREFIX}{date}{FILE_EXT}"
        src = os.path.join(SOURCE_DIR, filename)
        dest = os.path.join(TESTING_DIR, filename)
        shutil.move(src, dest)

    with open(os.path.join(TESTING_DIR, "testing_dates_note.txt"), "w") as f:
        f.write("--- testing data (20%) ---\n")
        f.write("\n".join(sorted(testing_dates)))

    for date in training_dates:
        filename = f"{FILE_PREFIX}{date}{FILE_EXT}"
        src = os.path.join(SOURCE_DIR, filename)
        dest = os.path.join(TRAINING_DIR, filename)
        shutil.move(src, dest)

    with open(os.path.join(TRAINING_DIR, "training_dates_note.txt"), "w") as f:
        f.write("--- training data (80%) ---\n")
        f.write("\n".join(sorted(training_dates)))

    print(
        f"{len(testing_dates)} files moved to testing and {len(training_dates)} files moved to training."
    )


if __name__ == "__main__":
    process_logs()
