import csv
from pathlib import Path


FINAL_DIR = Path("reports/final")
OUTPUT_PATH = Path("reports/merged/all_models_final_results.csv")


def main():
    files = sorted(FINAL_DIR.glob("*_final_results.csv"))

    if not files:
        print(f"No final result files found in: {FINAL_DIR}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    fieldnames = None

    for file_path in files:
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                print(f"Schema mismatch in file: {file_path}")
                print(f"Expected: {fieldnames}")
                print(f"Found:    {reader.fieldnames}")
                return

            for row in reader:
                all_rows.append(row)

    def sort_key(row):
        qid = row.get("question_id", "")
        try:
            qid_num = int(qid)
        except ValueError:
            qid_num = 10**9
        return (qid_num, row.get("model_name", ""), row.get("category", ""))

    all_rows.sort(key=sort_key)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Merged {len(files)} files into: {OUTPUT_PATH}")
    print(f"Total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()