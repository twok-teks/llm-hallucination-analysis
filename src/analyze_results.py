import csv
from collections import defaultdict
from pathlib import Path

INPUT_PATH = Path("reports/final_results.csv")


def safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    if not INPUT_PATH.exists():
        print(f"Missing file: {INPUT_PATH}")
        print("Run evaluate_results.py first.")
        return

    total = 0
    total_correct = 0

    correct_conf = []
    incorrect_conf = []

    category_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "all_conf": [],
        "correct_conf": [],
        "incorrect_conf": []
    })

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total += 1

            category = row["category"].strip()
            confidence = float(row["confidence"])
            is_correct = int(row["is_correct"])

            category_stats[category]["total"] += 1
            category_stats[category]["all_conf"].append(confidence)

            if is_correct == 1:
                total_correct += 1
                correct_conf.append(confidence)
                category_stats[category]["correct"] += 1
                category_stats[category]["correct_conf"].append(confidence)
            else:
                incorrect_conf.append(confidence)
                category_stats[category]["incorrect_conf"].append(confidence)

    accuracy = total_correct / total if total else 0.0
    hallucination_rate = 1.0 - accuracy

    avg_correct_conf = safe_mean(correct_conf)
    avg_incorrect_conf = safe_mean(incorrect_conf)

    print("\n===== OVERALL RESULTS =====")
    print(f"Total Questions: {total}")
    print(f"Correct Answers: {total_correct}")
    print(f"Incorrect Answers: {total - total_correct}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Hallucination Rate: {hallucination_rate:.3f}")
    print(f"Average Confidence (Correct):   {avg_correct_conf:.3f}")
    print(f"Average Confidence (Incorrect): {avg_incorrect_conf:.3f}")

    print("\n===== CATEGORY RESULTS =====")
    header = (
        f"{'Category':<15}"
        f"{'Total':>8}"
        f"{'Correct':>10}"
        f"{'Accuracy':>12}"
        f"{'Avg Conf':>12}"
        f"{'Correct Conf':>15}"
        f"{'Incorrect Conf':>17}"
    )
    print(header)
    print("-" * len(header))

    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        cat_total = stats["total"]
        cat_correct = stats["correct"]
        cat_accuracy = cat_correct / cat_total if cat_total else 0.0

        cat_avg_conf = safe_mean(stats["all_conf"])
        cat_avg_correct_conf = safe_mean(stats["correct_conf"])
        cat_avg_incorrect_conf = safe_mean(stats["incorrect_conf"])

        print(
            f"{category:<15}"
            f"{cat_total:>8}"
            f"{cat_correct:>10}"
            f"{cat_accuracy:>12.3f}"
            f"{cat_avg_conf:>12.3f}"
            f"{cat_avg_correct_conf:>15.3f}"
            f"{cat_avg_incorrect_conf:>17.3f}"
        )


if __name__ == "__main__":
    main()