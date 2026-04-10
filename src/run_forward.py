import csv
import os
import re
import argparse
from pathlib import Path

import torch

from load_model import load_model

PROMPTS_PATH = Path("prompts/factual_prompts.csv")
RAW_OUTPUT_DIR = Path("reports/raw")
MAX_NEW_TOKENS = 12


def load_questions(csv_path: Path):
    rows = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        filtered_lines = (
            line for line in f
            if line.strip() and not line.strip().startswith("#")
        )
        reader = csv.DictReader(filtered_lines)

        for idx, row in enumerate(reader, start=1):
            rows.append({
                "question_id": row.get("question_id", str(idx)).strip(),
                "question": row["question"].strip(),
                "answer": row["answer"].strip(),
                "category": row["category"].strip(),
            })

    return rows


def build_prompt(question: str) -> str:
    return (
        "Answer the following factual question in a short phrase.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def clean_model_answer(text: str) -> str:
    text = text.strip()
    text = text.split("\n")[0].strip()

    prefixes = [
        "Answer:",
        "answer:",
        "The answer is",
        "the answer is",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    text = re.sub(r"^[\s:,-]+", "", text)
    text = re.sub(r"[\s]+", " ", text).strip()
    return text


def generate_answer(tokenizer, model, device, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = generated[0][input_len:]
    answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    answer_text = clean_model_answer(answer_text)

    return answer_text, generated[0], input_len


def compute_confidence(model, full_sequence, input_len: int):
    full_sequence = full_sequence.unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids=full_sequence)

    logits = outputs.logits
    token_probs = []

    for i in range(input_len, full_sequence.shape[1]):
        prev_pos = i - 1
        token_id = full_sequence[0, i].item()

        probs = torch.softmax(logits[0, prev_pos], dim=-1)
        token_prob = probs[token_id].item()
        token_probs.append(token_prob)

    if not token_probs:
        return 0.0

    return sum(token_probs) / len(token_probs)


def ensure_output_dir(path: Path):
    os.makedirs(path.parent, exist_ok=True)


def build_output_path(model_key: str) -> Path:
    return RAW_OUTPUT_DIR / f"{model_key}_raw_results.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["tinyllama", "phi3", "mistral", "llama3"],
        help="Which model to run",
    )
    parser.add_argument(
        "--prompts",
        default=str(PROMPTS_PATH),
        help="Path to prompt CSV",
    )
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    output_path = build_output_path(args.model)
    ensure_output_dir(output_path)

    tokenizer, model, device, resolved_model_name = load_model(args.model)
    rows = load_questions(prompts_path)

    print(f"Loaded {len(rows)} prompts.")
    print(f"Running experiment with: {args.model} ({resolved_model_name})")
    print()

    results = []

    for idx, row in enumerate(rows, start=1):
        question_id = row["question_id"]
        question = row["question"]
        ground_truth = row["answer"]
        category = row["category"]

        prompt = build_prompt(question)
        model_answer, full_sequence, input_len = generate_answer(
            tokenizer, model, device, prompt
        )
        confidence = compute_confidence(model, full_sequence.to(device), input_len)

        results.append({
            "question_id": question_id,
            "question": question,
            "ground_truth_answer": ground_truth,
            "category": category,
            "model_name": args.model,
            "model_answer": model_answer,
            "confidence": f"{confidence:.6f}",
        })

        print(f"[{idx}/{len(rows)}] {question}")
        print(f"  Ground truth: {ground_truth}")
        print(f"  Model answer: {model_answer}")
        print(f"  Confidence:   {confidence:.6f}")
        print()

    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "question_id",
            "question",
            "ground_truth_answer",
            "category",
            "model_name",
            "model_answer",
            "confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()