import csv
import os
import re
from pathlib import Path

import torch

from load_model import load_model

PROMPTS_PATH = Path("prompts/factual_prompts.csv")
OUTPUT_PATH = Path("reports/raw_results.csv")
MAX_NEW_TOKENS = 12


def load_questions(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        filtered_lines = (
            line for line in f
            if line.strip() and not line.strip().startswith("#")
        )
        reader = csv.DictReader(filtered_lines)
        for row in reader:
            rows.append({
                "question": row["question"].strip(),
                "answer": row["answer"].strip(),
                "category": row["category"].strip()
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
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = generated[0][input_len:]
    answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    answer_text = clean_model_answer(answer_text)

    return answer_text, generated[0], input_len


def compute_confidence(model, full_sequence, input_len: int):
    """
    Average probability of generated tokens.
    """
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


def main():
    ensure_output_dir(OUTPUT_PATH)

    tokenizer, model, device = load_model()
    rows = load_questions(PROMPTS_PATH)

    print(f"Loaded {len(rows)} prompts.")
    print("Running experiment...\n")

    results = []

    for idx, row in enumerate(rows, start=1):
        question = row["question"]
        ground_truth = row["answer"]
        category = row["category"]

        prompt = build_prompt(question)
        model_answer, full_sequence, input_len = generate_answer(
            tokenizer, model, device, prompt
        )
        confidence = compute_confidence(model, full_sequence.to(device), input_len)

        results.append({
            "question": question,
            "ground_truth_answer": ground_truth,
            "category": category,
            "model_answer": model_answer,
            "confidence": f"{confidence:.6f}"
        })

        print(f"[{idx}/{len(rows)}] {question}")
        print(f"  Ground truth: {ground_truth}")
        print(f"  Model answer: {model_answer}")
        print(f"  Confidence:   {confidence:.6f}\n")

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "question",
            "ground_truth_answer",
            "category",
            "model_answer",
            "confidence"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Done. Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()