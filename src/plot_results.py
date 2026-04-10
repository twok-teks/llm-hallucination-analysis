import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

INPUT_PATH = Path("reports/final_results.csv")
OUTPUT_DIR = Path("reports/plots")


def main():
    if not INPUT_PATH.exists():
        print(f"Missing file: {INPUT_PATH}")
        print("Run evaluate_results.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    correct_conf = []
    incorrect_conf = []

    category_total = defaultdict(int)
    category_correct = defaultdict(int)

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            category = row["category"].strip()
            confidence = float(row["confidence"])
            is_correct = int(row["is_correct"])

            category_total[category] += 1

            if is_correct == 1:
                category_correct[category] += 1
                correct_conf.append(confidence)
            else:
                incorrect_conf.append(confidence)

    # Plot 1: Confidence histogram
    plt.figure(figsize=(8, 5))
    plt.hist(correct_conf, bins=20, alpha=0.7, label="Correct")
    plt.hist(incorrect_conf, bins=20, alpha=0.7, label="Incorrect")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Confidence Distribution: Correct vs Incorrect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confidence_histogram.png")
    plt.close()

    # Plot 2: Confidence boxplot
    plt.figure(figsize=(6, 5))
    plt.boxplot([correct_conf, incorrect_conf], tick_labels=["Correct", "Incorrect"])
    plt.ylabel("Confidence")
    plt.title("Confidence Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confidence_boxplot.png")
    plt.close()

    # Plot 3: Category accuracy
    categories = sorted(category_total.keys())
    accuracies = [
        category_correct[c] / category_total[c] if category_total[c] else 0.0
        for c in categories
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(categories, accuracies)
    plt.xlabel("Category")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_accuracy.png")
    plt.close()

    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()