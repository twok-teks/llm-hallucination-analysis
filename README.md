# 🧠 LLM Hallucination Analysis & Detection

<p align="center">
  <b>Understanding when LLMs fail... and learning how to catch it.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange"/>
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow"/>
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-green"/>
</p>

---

## 🚀 Overview

Large Language Models (LLMs) can produce **confident but incorrect answers**, known as *hallucinations*.

This project builds a **full evaluation + ML pipeline** to:

* 📊 Analyze hallucination behavior across multiple models
* 📉 Study confidence vs correctness
* 🤖 Train a machine learning model to detect hallucinations

---

## ❗ Problem

LLMs:

* ❌ can be wrong
* ⚠️ can sound extremely confident
* 🤯 give no built-in signal they’re hallucinating

👉 This project answers:

> Can we **detect hallucinations automatically** using model outputs?

---

## 🧪 Models Evaluated

| Model      | Size  | Type            |
| ---------- | ----- | --------------- |
| TinyLlama  | 1.1B  | Lightweight     |
| Phi-3 Mini | ~3.8B | Efficient       |
| Mistral    | 7B    | Strong baseline |
| LLaMA 3.1  | 8B    | Advanced        |

---

## 📂 Project Structure

```bash
llm-hallucination-analysis/
│
├── prompts/
│   └── factual_prompts.csv
│
├── src/
│   ├── run_forward.py        # Run models
│   ├── evaluate_results.py  # Compare with ground truth
│   ├── merge_results.py     # Merge datasets
│   ├── analyze_results.py   # Metrics
│   ├── plot_results.py      # Visualizations
│   └── train_detector.py    # ML model
│
├── reports/
│   ├── raw/       # Raw model outputs
│   ├── final/     # Evaluated outputs
│   ├── merged/    # Combined dataset
│   ├── metrics/   # CSV + JSON metrics
│   └── plots/     # All graphs
│
└── README.md
```

---

## 📊 Sample Visualizations

### 🔹 Confidence vs Correctness

![Image](https://images.openai.com/static-rsc-4/Lka1xeIz5WbWWtO3_kD-2Bxlrgxwi8e_6S00FaxKgiLAgY_xVWbKNeMfzb-e2kZBx-aOJ0PRNWy8fXVY6vCSs9GhU8qy-iAL8frFL-5pTS27rQAtZcl-UUdWGu7d-WKWVMbezYSUp63OE06Yllu6pSV4J9ZcLYjZLZaJJoJqyHZjkk8F_eN8ci7R9UTx29R-?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/Iz1zNcKjNpez1RqxlUaXFOzl6gTrNJHerLlnD3y_Z5P1LH8YedwlNK368a7jq7m1kBfUuz0inzCuprY6gzVatrulkBq7BzSfxa2OQtg5CwC4dZ4g4vRpof6TAv-n2FBmBb6jkMv0NC0IdjqZGmJnP2KauHz1FO8y13LT1MUoxG44yVkwu6xWal38C0b2hLqm?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/dQDrRC_N01YXlvQpvOpcc6vxE5sADubWW7cJSpD2j7q7AFuUSZXjJ1k4hC33p-afQ4hwVg8uRe7dywijkElH6EChS89n2xfcMLyB6Dm3F74Te-awcBHHP9NJ9kTPQX4U1N2JwFKMFvgm4BIsBszr9pmfpBY99akPQPnHB9EpGhN7ORQGLcgN9FIBPHib2j2T?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/GWGobaCqsiW3RDjopN3FrRHZeBq8I9uN4kRKRj4LrXp682fAmOxT8YJz6EbgeKCCdfFIbEVQ8C7UZU-jPmZW65b_KqFbKGa_y7unlJuGAC4DqYWcuwUvAMjqBZ18QjA6mfmlrOO8UyqoMGrMpB-55I_Yv54DYnj3545-QVpsRBGcvqjU_GUW_k3P3RfBQbZO?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/Zp7MLu_BGWKXr0HkDEBE1NRWPZHBe_kUW7Jl2ko_PQVcj6_TNluBqZNwhWZDmgi0OX6lmUFLxP34C6t18hCSBcjt6t49kPX31TGreETAnhhMsYkBPJ4OIyrO0OotKPXXL8WlPL-itEDZnWI7QVZs-vkBhqj4ROjwLwd03G5b5yPq-DteUF6DDrqX3OYziPQu?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/2Z0sCtahOqGzAjNYuAZZkG9DAWMVt5rKik9RL9zx9p6Bx841zrHZv_xcrwxM48hHxXn0aOM_HwYa4hiEaVWpWhURAJREm5VQJYMNP7wFmebi3xfwfZoprhl1M24SFzmprHX1S-ux2FbnuiYWg0mx-ft9WcICH-U5LF5ogTaQMrrPzVTBtImVlUkzE_QiT-Ws?purpose=fullsize)

---

### 🔹 Model Comparison

![Image](https://images.openai.com/static-rsc-4/eT9U3l5reEbmK9h7823b6VQ9v-xMhjuaIO_R_JQw7-xRYOOS4D1rI7eQ9-yDs9rmOssZQwbkTIKGas-B3QLpVB7I7Ok1CK-ty1LjEjoxC4maoHFM9X4Gm54qXASbPQPkzrxQpd8UV4qfi6BNjIJdOZxIpA_kfFxvQF7DokMoPxs4dZhoEHHFsKfrorZChbS8?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/FfA2RTtR6lWBHlBG8hR8YL3YhmkQLGkS4oP4yUb-Cqa6zBn7HecBPwqwLMNrJfTziEgFXsGes92xgitPMrSHRmVdRX32M7AFEd_-rPfV0wH1R3GT7qhkQPxwBAq1XHLTOdQbFR4wH083coI_mVUa7nZBs8dvOE21LhUVC2_OpEtZwtxmRa-dGxLHkVUjlrvV?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/KTIwc7TwlsuRR9rxDh0zCdqaxXvti2o2ToVwMBiZ4GDCvYC1yPV-yfv76vLoYQL8AUKu4_fp5EOWGcn5mqhon-z5RdK8AsDA2OvBJ7hvLvJ8wbVXVVQNcBKoLDXSo1bw8kwB5yEOyBF5DU9n3zewlPRK0QWZkwCvcuswgbSQU8IRF0gpDwY-YvPstkqxka7f?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/wBsWKHPzoQYnGxDztF-sZx7UAkHfVPI7pA2OX_raN7yKMs00RWxdeV2_5rc6lsre7LuS72oI9_tTyK5kHPQttR4w2zqr2DxMbZ1icBfjZpF3Pfr4Vxx3FF8V0jXMWo-XUEcqsHvwqRJvsdDEpxS_pHocXl0lZPE5yy-SIeBbPC81sG51DSH_ttVFVicsrjxP?purpose=fullsize)

---

### 🔹 Heatmap Analysis

![Image](https://images.openai.com/static-rsc-4/lVrS6IXGPUo_kLJwEdo8WxNZvF2psA6-Zy_t5WWCRIasfzv0rpKWN64bbmSwW9Rsh3PuiKZ29rqQuJeRW3C0lkEeZENoxna22bDK606FAe0u8De1fKOrgWdvQMRlUy0HzbADvhp7n_dj_UqPlLwcrQyqnrD6SvxhvKRqTZHLDHaIOmmOApzNVG2AGd7fw9Oo?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/hzcfdYZD48Q88zHpiPnw7i3YuKIR2YIK5tNQVDUP-s1o7qk-qmu5Zf5ErOy3uaWz0y4rDk92vP77pnMOBDzJcDHC70ecICJEGoPM2UzX9cHn1SuNtqb4ftYFHrsc_YlnH_le17PIUeoYy1R1DBxPJhPBngjSO8Gj10I2Yc5CeBfQFH3EOuQ8mrt41Q9CzUAT?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/7Xse9zBV1M2OwMEZhap583sVFZOOrVu5aFKaGMFta1RgxE1lXHAtZM1KxcjDnP0YotFEb3AWbgKLXDqIcr0H_VSserhTqvvOSL3JO8BwNKnv4dKlcgqjMHoc_e__oETu6tTw98X_DUA8keY_hBkUpXvcmq_y6Ef4AjS8_4tQGY2ko2zjwpp3KJ-8TRQFoWUl?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/aRiziMcJI46L-Ox93zDAmCm839yzFrYkJ8NLK0DFG70YpzdyBBbmiBHi7gIFsQbkEQmuix5mNk-r21C_XjVSWwWXd1DmVn6iG2bXJiTU1ejdNnNdFPDQW_YcLIImfqqQwv3Ld405EtLAnwrDUFEyinY5u0stKWusVnHIM7LCX9RR9dMJiQiKT4TP0d-xxOSI?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/u7hGCBmWP5lfAeuFeQ5YF8E1VCZjREuQPc7PcN2hj1L9BfNt40P0mO2U8P1hgKhW21ukt0Zk6h0fR9bYzNQQku59vrHZLQNmJ9ux9V8eU18n22GWhYgnrY7s9BYHPvfsXMZ5kG5WgEyg9AXhmUyk2weJ42gru2LylyJ_ExSdbZGag1yAuq8j6q2WVTjkGwWq?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/FSqVSp3AVW8LVKPzCEX8aifckWVmTLrb4zhZ59151Sr9BBUXoFDrjVvSMWOP00uF9hASi_zLJVYo4NQqekWd8WRf8HwkwaqRSW1qeKePbTt9Zv2fD7vAO5lHTwKcuQlf64FzqCi8RwVXfeQUdTFJSHHvnYa6aj-A3a0QyUQN2ZDwppLtgly7ORoLQlV3uSLY?purpose=fullsize)

---

## 🧠 Methodology

### 1. Prompting

Each model is asked:

```text
Answer the following factual question in a short phrase.
Question: <question>
Answer:
```

### 2. Confidence Score

We compute:

> Average probability of generated tokens

### 3. Labeling

| Condition                   | Label            |
| --------------------------- | ---------------- |
| Correct answer              | ✅ Correct        |
| Incorrect answer            | ❌ Incorrect      |
| Incorrect + high confidence | 🚨 Hallucination |

---

## 🤖 ML Hallucination Detector

### Features

* confidence
* question length
* answer length
* category
* model name

### Models

* Logistic Regression
* Random Forest

### Outputs

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Feature Importance

---

## 📈 Key Insights

* 🔥 Larger models still hallucinate
* 📉 Confidence ≠ correctness
* 🧠 Some domains are more error-prone
* 🤖 ML models can detect hallucinations with meaningful accuracy

---

# ⚙️ How To Run Locally (Step-by-Step)

## 1. Clone Repo

```bash
git clone https://github.com/yourusername/llm-hallucination-analysis.git
cd llm-hallucination-analysis
```

---

## 2. Create Virtual Environment

### Windows (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. (Optional) Fix GPU Support ⚠️

If using RTX 50-series GPU:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 5. Run Models

Start small:

```bash
python src/run_forward.py --model tinyllama
```

Then:

```bash
python src/run_forward.py --model phi3
python src/run_forward.py --model mistral
python src/run_forward.py --model llama3
```

---

## 6. Evaluate Outputs

```bash
python src/evaluate_results.py --input reports/raw/tinyllama_raw_results.csv --output reports/final/tinyllama_final_results.csv --model tinyllama

python src/evaluate_results.py --input reports/raw/phi3_raw_results.csv --output reports/final/phi3_final_results.csv --model phi3

python src/evaluate_results.py --input reports/raw/mistral_raw_results.csv --output reports/final/mistral_final_results.csv --model mistral

python src/evaluate_results.py --input reports/raw/llama3_raw_results.csv --output reports/final/llama3_final_results.csv --model llama3
```

---

## 7. Merge Data

```bash
python src/merge_results.py
```

---

## 8. Generate Metrics + Graphs

```bash
python src/analyze_results.py
python src/plot_results.py
```

---

## 9. Train Hallucination Detector

```bash
python src/train_detector.py
```

---

## 📁 Output Files

After running:

* 📊 `reports/plots/` → all graphs
* 📄 `reports/metrics/` → performance summaries
* 📦 `reports/merged/` → dataset for ML

---

## 🔬 Future Work

* Add GPT / Claude models
* Improve feature engineering
* Use token-level entropy
* Train deep learning detectors
* Scale dataset to 1000+ questions

---

## 🙌 Author

**Khanh Van**
UT Dallas → UT Austin MSCS

---

## ⭐ Final Thought

> LLMs don’t just make mistakes —
> they make *confident* mistakes.

Detecting them is the first step toward **trustworthy AI**.
