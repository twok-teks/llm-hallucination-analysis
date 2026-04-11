import csv
from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path("reports/merged/all_models_final_results.csv")
MODEL_SUMMARY_PATH = Path("reports/metrics/model_summary.csv")
CATEGORY_SUMMARY_PATH = Path("reports/metrics/category_summary.csv")
CATEGORY_ONLY_SUMMARY_PATH = Path("reports/metrics/category_only_summary.csv")


def safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    if not INPUT_PATH.exists():
        print(f"Missing file: {INPUT_PATH}")
        return

    MODEL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "hallucination": 0,
        "refusal": 0,
        "correct_conf": [],
        "incorrect_conf": [],
        "hallucinated_conf": [],
        "all_conf": [],
    })

    category_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "hallucination": 0,
        "refusal": 0,
    })

    model_category_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "hallucination": 0,
        "refusal": 0,
    })

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_cols = {
            "model_name", "category", "is_correct",
            "hallucination", "refusal", "confidence"
        }
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            print(f"Missing required columns in {INPUT_PATH}: {sorted(missing)}")
            return

        for row in reader:
            model = row["model_name"].strip()
            category = row["category"].strip()
            is_correct = int(row["is_correct"])
            hallucination = int(row["hallucination"])
            refusal = int(row["refusal"])
            confidence = float(row["confidence"])

            model_stats[model]["total"] += 1
            model_stats[model]["correct"] += is_correct
            model_stats[model]["hallucination"] += hallucination
            model_stats[model]["refusal"] += refusal
            model_stats[model]["all_conf"].append(confidence)

            if is_correct == 1:
                model_stats[model]["correct_conf"].append(confidence)
            else:
                model_stats[model]["incorrect_conf"].append(confidence)

            if hallucination == 1:
                model_stats[model]["hallucinated_conf"].append(confidence)

            category_stats[category]["total"] += 1
            category_stats[category]["correct"] += is_correct
            category_stats[category]["hallucination"] += hallucination
            category_stats[category]["refusal"] += refusal

            key = (model, category)
            model_category_stats[key]["total"] += 1
            model_category_stats[key]["correct"] += is_correct
            model_category_stats[key]["hallucination"] += hallucination
            model_category_stats[key]["refusal"] += refusal

    with MODEL_SUMMARY_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model_name",
            "total",
            "correct",
            "incorrect",
            "accuracy",
            "hallucination_rate",
            "refusal_rate",
            "avg_confidence",
            "avg_correct_confidence",
            "avg_incorrect_confidence",
            "avg_hallucinated_confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print("\n===== MODEL SUMMARY =====")
        for model in sorted(model_stats.keys()):
            stats = model_stats[model]
            total = stats["total"]
            correct = stats["correct"]
            incorrect = total - correct

            row = {
                "model_name": model,
                "total": total,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": round(correct / total if total else 0.0, 4),
                "hallucination_rate": round(stats["hallucination"] / total if total else 0.0, 4),
                "refusal_rate": round(stats["refusal"] / total if total else 0.0, 4),
                "avg_confidence": round(safe_mean(stats["all_conf"]), 4),
                "avg_correct_confidence": round(safe_mean(stats["correct_conf"]), 4),
                "avg_incorrect_confidence": round(safe_mean(stats["incorrect_conf"]), 4),
                "avg_hallucinated_confidence": round(safe_mean(stats["hallucinated_conf"]), 4),
            }
            writer.writerow(row)
            print(row)

    with CATEGORY_SUMMARY_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model_name",
            "category",
            "total",
            "correct",
            "accuracy",
            "hallucination_rate",
            "refusal_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print("\n===== MODEL × CATEGORY SUMMARY =====")
        for (model, category) in sorted(model_category_stats.keys()):
            stats = model_category_stats[(model, category)]
            total = stats["total"]

            row = {
                "model_name": model,
                "category": category,
                "total": total,
                "correct": stats["correct"],
                "accuracy": round(stats["correct"] / total if total else 0.0, 4),
                "hallucination_rate": round(stats["hallucination"] / total if total else 0.0, 4),
                "refusal_rate": round(stats["refusal"] / total if total else 0.0, 4),
            }
            writer.writerow(row)

    with CATEGORY_ONLY_SUMMARY_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "category",
            "total",
            "correct",
            "accuracy",
            "hallucination_rate",
            "refusal_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print("\n===== CATEGORY ONLY SUMMARY =====")
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            total = stats["total"]

            row = {
                "category": category,
                "total": total,
                "correct": stats["correct"],
                "accuracy": round(stats["correct"] / total if total else 0.0, 4),
                "hallucination_rate": round(stats["hallucination"] / total if total else 0.0, 4),
                "refusal_rate": round(stats["refusal"] / total if total else 0.0, 4),
            }
            writer.writerow(row)

    print(f"\nSaved: {MODEL_SUMMARY_PATH}")
    print(f"Saved: {CATEGORY_SUMMARY_PATH}")
    print(f"Saved: {CATEGORY_ONLY_SUMMARY_PATH}")


if __name__ == "__main__":
    main()