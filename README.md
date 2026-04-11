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

This project builds a **full evaluation + machine learning pipeline** to:

- 📊 Analyze hallucination behavior across multiple LLMs  
- 📉 Study confidence vs correctness vs refusal behavior  
- 🤖 Train a model to detect hallucinated responses automatically  

---

## ❗ Problem

LLMs:

- ❌ Can be wrong  
- ⚠️ Can sound extremely confident  
- 🤯 Provide no built-in hallucination signal  

👉 This project answers:

> Can we **automatically detect hallucinated responses** using model outputs?

---

## 🧪 Models Evaluated

| Model      | Size  | Type            |
| ---------- | ----- | --------------- |
| TinyLlama  | 1.1B  | Lightweight     |
| Phi-3 Mini | ~3.8B | Efficient       |
| Mistral    | 7B    | Strong baseline |
| LLaMA 3.1  | 8B    | Advanced        |

---

## 🧠 Key Idea

Instead of predicting hallucination from prompts, this project focuses on:

**Post-response detection**

Prompt → LLM → Answer → Detector → Hallucination?

---

## 🧪 Dataset Design

Prompts are divided into 4 groups:

| Type        | Purpose |
|------------|--------|
| Easy        | Basic factual correctness |
| Hard        | Challenging knowledge |
| Trap        | False / invalid premises |
| Adversarial | Forces confident answering |

---

## 🧠 Labeling Strategy

| Condition                        | Label |
|--------------------------------|------|
| Correct answer                  | ✅ Correct |
| Incorrect + refusal             | ✅ Not hallucination |
| Incorrect + no refusal          | 🚨 Hallucination |

---

## 📊 Results Summary

### 🔥 Model Behavior

| Model      | Accuracy | Hallucination Rate |
|------------|----------|-------------------|
| LLaMA3     | ~96%     | ~2.7%             |
| Mistral    | ~95%     | ~4.5%             |
| Phi-3      | ~93%     | ~7%               |
| TinyLlama  | ~56%     | ~40%              |

---

### 🤖 Hallucination Detector Performance

**Random Forest (Best Model)**

- Accuracy: **93.3%**
- Precision: **70%**
- Recall: **87.5%**
- F1 Score: **0.78**
- ROC AUC: **0.96**
- PR AUC: **0.87**

👉 Strong ability to detect hallucinations while keeping false positives low.

---

## 🧠 Feature Importance

Top signals:

1. Refusal behavior (most important)
2. Model type
3. Confidence score
4. Answer length
5. Prompt category

---

## 🔍 Key Insights

- 🔥 Larger models still hallucinate  
- 📉 Confidence ≠ correctness  
- 🧠 Refusal is a strong safety mechanism  
- ⚠️ Small models hallucinate significantly more  
- 🤖 ML models can reliably detect hallucinations  

---

## ⚙️ Pipeline

run_forward.py        → Generate model outputs  
evaluate_results.py   → Label correctness + hallucination  
merge_results.py      → Combine datasets  
analyze_results.py    → Metrics  
plot_results.py       → Visualizations  
train_detector.py     → ML model  

---

## ⚙️ How To Run Locally

### 1. Clone Repo

git clone https://github.com/yourusername/llm-hallucination-analysis.git  
cd llm-hallucination-analysis  

---

### 2. Setup Environment

python -m venv .venv  
.venv\Scripts\activate  
pip install -r requirements.txt  

---

### 3. GPU Fix (RTX 50 Series)

pip uninstall -y torch torchvision torchaudio  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  

---

### 4. Run Models

python src/run_forward.py --model tinyllama  
python src/run_forward.py --model phi3  
python src/run_forward.py --model mistral  
python src/run_forward.py --model llama3  

---

### 5. Evaluate + Analyze

python src/evaluate_results.py ...  
python src/merge_results.py  
python src/analyze_results.py  
python src/plot_results.py  

---

### 6. Train Detector

python src/train_detector.py  

---

## 📁 Outputs

| Folder | Description |
|--------|------------|
| reports/raw | Raw outputs |
| reports/final | Labeled results |
| reports/merged | Dataset |
| reports/metrics | Metrics |
| reports/plots | Graphs |

---

## 🔬 Future Work

- Add GPT / Claude APIs  
- Use token-level entropy / logits  
- Build real-time detection API  
- Expand dataset (1000+ prompts)  

---

## 🙌 Author

**Khanh Van**  
UT Dallas → UT Austin MSCS  

---

## ⭐ Final Thought

LLMs don’t just make mistakes —  
they make *confident* mistakes  

Detecting them is the first step toward **trustworthy AI**


---

## 🧪 Completed Improvements & Final Pipeline

### ✅ Improved Labeling Logic
- Hallucination = incorrect + NOT refusal  
- Distinguishes safe uncertainty vs fabricated answers  

### ✅ Dataset Upgrade
- Balanced prompt categories  
- Added `question_validity` and `refusal`  
- Clean ground truth alignment  

### ✅ Feature Engineering
- refusal  
- length_ratio  
- confidence × answer_length  
- low_confidence_flag  

### 🤖 Final Detector Performance
- Accuracy: 93.3%  
- Recall: 87.5%  
- Precision: 70%  
- ROC AUC: 0.96  

### 📊 Confusion Matrix
TN: 146 | FP: 9  
FN: 3   | TP: 21  

### 📈 Key Findings
- Models are confident when wrong  
- Refusal is strongest anti-hallucination signal  
- Smaller models hallucinate more  

### ⚡ Infrastructure Improvement
- Enabled GPU (CUDA 12.8)  
- Used float16 + device_map auto  
- Improved runtime performance  
