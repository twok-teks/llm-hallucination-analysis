import csv
import re
from pathlib import Path

RAW_PATH = Path("reports/raw_results.csv")
FINAL_PATH = Path("reports/final_results.csv")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def looks_like_rejection(text: str) -> bool:
    text = normalize(text)

    rejection_phrases = [
        "none",
        "no such",
        "does not exist",
        "doesnt exist",
        "not real",
        "fictional",
        "mythological",
        "there is no",
        "no real",
        "invalid",
        "not a real",
        "hypothetical",
        "imaginary",
        "n a",
        "na",
    ]

    return any(phrase in text for phrase in rejection_phrases)


def is_correct_answer(ground_truth: str, model_answer: str) -> int:
    gt = normalize(ground_truth)
    pred = normalize(model_answer)

    # For trick/impossible questions:
    # mark correct only if the model rejects the premise
    if gt == "none":
        return 1 if looks_like_rejection(model_answer) else 0

    # Exact match
    if pred == gt:
        return 1

    # Bidirectional contains
    if gt in pred or pred in gt:
        return 1

    return 0


def main():
    if not RAW_PATH.exists():
        print(f"Missing file: {RAW_PATH}")
        return

    rows = []

    with RAW_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            question = row["question"].strip()
            ground_truth = row["ground_truth_answer"].strip()
            category = row["category"].strip()
            model_answer = row["model_answer"].strip()
            confidence = row["confidence"].strip()

            correct = is_correct_answer(ground_truth, model_answer)

            rows.append({
                "question": question,
                "ground_truth_answer": ground_truth,
                "category": category,
                "model_answer": model_answer,
                "confidence": confidence,
                "is_correct": correct
            })

    with FINAL_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "question",
            "ground_truth_answer",
            "category",
            "model_answer",
            "confidence",
            "is_correct"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {FINAL_PATH}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()