import csv
import re
import argparse
from pathlib import Path


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
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
        "not applicable",
        "not applicable.",
        "not",
        "no"
    ]

    return any(phrase in text for phrase in rejection_phrases)


def is_correct_answer(ground_truth: str, model_answer: str) -> int:
    gt = normalize(ground_truth)
    pred = normalize(model_answer)

    if gt == "none":
        return 1 if looks_like_rejection(model_answer) else 0

    if pred == gt:
        return 1

    if gt in pred or pred in gt:
        return 1

    return 0


def infer_model_name(input_path: Path) -> str:
    name = input_path.stem.lower()
    name = name.replace("_raw_results", "")
    name = name.replace("_results", "")
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw results CSV")
    parser.add_argument("--output", required=True, help="Path to final results CSV")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--hallucination-threshold", type=float, default=0.65)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Missing file: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    model_name = args.model if args.model else infer_model_name(input_path)

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader, start=1):
            question = row["question"].strip()
            ground_truth = row["ground_truth_answer"].strip()
            category = row["category"].strip()
            model_answer = row["model_answer"].strip()
            confidence = float(row["confidence"].strip())

            question_id = row.get("question_id", str(idx)).strip()
            row_model_name = row.get("model_name", "").strip() or model_name

            correct = is_correct_answer(ground_truth, model_answer)
            hallucination = 1 if (correct == 0 and confidence >= args.hallucination_threshold) else 0

            rows.append({
                "question_id": question_id,
                "question": question,
                "ground_truth_answer": ground_truth,
                "category": category,
                "model_name": row_model_name,
                "model_answer": model_answer,
                "confidence": confidence,
                "is_correct": correct,
                "hallucination": hallucination,
                "question_length": len(question.split()),
                "answer_length": len(model_answer.split()),
            })

    fieldnames = [
        "question_id",
        "question",
        "ground_truth_answer",
        "category",
        "model_name",
        "model_answer",
        "confidence",
        "is_correct",
        "hallucination",
        "question_length",
        "answer_length",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {output_path}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()