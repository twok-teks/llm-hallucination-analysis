import csv
import re
import argparse
from pathlib import Path


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def looks_like_rejection(text: str) -> bool:
    text = normalize(text)

    rejection_phrases = [
        "i dont know",
        "i do not know",
        "unknown",
        "cannot be determined",
        "cant be determined",
        "cannot answer",
        "not enough information",
        "does not exist",
        "doesnt exist",
        "not real",
        "fictional",
        "mythical",
        "mythological",
        "imaginary",
        "there is no",
        "no such",
        "invalid premise",
        "this premise is false",
        "not a real",
        "hypothetical",
        "no evidence",
        "no verified",
        "not possible to know exactly",
        "cannot be known exactly",
        "there is no exact",
        "no",
        "none",
        "no such",
        "not applicable",
        "na",
        "n a",
        "unavailable",
        "not",
        "false"
    ]

    return any(phrase in text for phrase in rejection_phrases)


def is_correct_answer(ground_truth: str, model_answer: str, question_validity: str) -> int:
    gt = normalize(ground_truth)
    pred = normalize(model_answer)
    validity = normalize(question_validity)

    # For invalid or inherently unanswerable prompts,
    # the correct behavior is to reject / refuse the premise.
    if validity in {"invalid_premise", "unanswerable_exact"} or gt == "none":
        return 1 if looks_like_rejection(model_answer) else 0

    if not pred:
        return 0

    if pred == gt:
        return 1

    # Allow exact answer embedded inside explanation
    if gt and gt in pred:
        return 1

    return 0


def infer_model_name(input_path: Path) -> str:
    name = input_path.stem.lower()
    name = name.replace("_raw_results", "")
    name = name.replace("_results", "")
    return name


def infer_question_validity(row: dict) -> str:
    value = row.get("question_validity", "").strip().lower()
    if value:
        return value

    gt = normalize(row.get("ground_truth_answer", ""))
    category = row.get("category", "").strip().lower()

    if gt == "none":
        return "invalid_premise"

    if category in {"trap"}:
        return "invalid_premise"

    if category in {"adversarial"}:
        return "unanswerable_exact"

    return "valid"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw results CSV")
    parser.add_argument("--output", required=True, help="Path to final results CSV")
    parser.add_argument("--model", default=None, help="Override model name")
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

        required_cols = {
            "question",
            "ground_truth_answer",
            "category",
            "model_answer",
            "confidence",
        }
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            print(f"Missing required columns in {input_path}: {sorted(missing)}")
            return

        for idx, row in enumerate(reader, start=1):
            question = row["question"].strip()
            ground_truth = row["ground_truth_answer"].strip()
            category = row["category"].strip().lower()
            question_validity = infer_question_validity(row)
            model_answer = row["model_answer"].strip()
            confidence = float(row["confidence"].strip())

            question_id = row.get("question_id", str(idx)).strip()
            row_model_name = row.get("model_name", "").strip() or model_name

            refusal = 1 if looks_like_rejection(model_answer) else 0
            correct = is_correct_answer(ground_truth, model_answer, question_validity)

            # Hallucination = incorrect answer that does NOT honestly refuse
            hallucination = 1 if (correct == 0 and refusal == 0) else 0

            rows.append({
                "question_id": question_id,
                "question": question,
                "ground_truth_answer": ground_truth,
                "category": category,
                "question_validity": question_validity,
                "model_name": row_model_name,
                "model_answer": model_answer,
                "confidence": confidence,
                "is_correct": correct,
                "refusal": refusal,
                "hallucination": hallucination,
                "question_length": len(question.split()),
                "answer_length": len(model_answer.split()),
            })

    fieldnames = [
        "question_id",
        "question",
        "ground_truth_answer",
        "category",
        "question_validity",
        "model_name",
        "model_answer",
        "confidence",
        "is_correct",
        "refusal",
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