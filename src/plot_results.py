import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("reports/merged/all_models_final_results.csv")
MODEL_SUMMARY_PATH = Path("reports/metrics/model_summary.csv")
CATEGORY_SUMMARY_PATH = Path("reports/metrics/category_summary.csv")
OUTPUT_DIR = Path("reports/plots/combined")
PER_MODEL_DIR = Path("reports/plots/per_model")


def save_heatmap(df, value_col, title, filename):
    pivot = df.pivot(index="model_name", columns="category", values=value_col)

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label=value_col.replace("_", " ").title())
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def save_per_model_histograms(df):
    os.makedirs(PER_MODEL_DIR, exist_ok=True)

    for model_name, group in df.groupby("model_name"):
        correct_conf = group[group["is_correct"] == 1]["confidence"]
        halluc_conf = group[group["hallucination"] == 1]["confidence"]

        plt.figure(figsize=(8, 5))
        plt.hist(correct_conf, bins=20, alpha=0.7, label="Correct")
        plt.hist(halluc_conf, bins=20, alpha=0.7, label="Hallucinated")
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.title(f"Confidence Distribution: {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PER_MODEL_DIR / f"{model_name}_confidence_histogram.png")
        plt.close()


def save_confidence_boxplot(df):
    correct_conf = df[df["is_correct"] == 1]["confidence"]
    halluc_conf = df[df["hallucination"] == 1]["confidence"]

    plt.figure(figsize=(7, 5))
    plt.boxplot([correct_conf, halluc_conf], tick_labels=["Correct", "Hallucinated"])
    plt.ylabel("Confidence")
    plt.title("Confidence: Correct vs Hallucinated")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confidence_boxplot_correct_vs_hallucinated.png")
    plt.close()


def save_confidence_violin(df):
    correct_conf = df[df["is_correct"] == 1]["confidence"]
    halluc_conf = df[df["hallucination"] == 1]["confidence"]

    plt.figure(figsize=(7, 5))
    plt.violinplot([correct_conf, halluc_conf], showmeans=True, showmedians=True)
    plt.xticks([1, 2], ["Correct", "Hallucinated"])
    plt.ylabel("Confidence")
    plt.title("Confidence Violin Plot: Correct vs Hallucinated")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confidence_violin_correct_vs_hallucinated.png")
    plt.close()


def save_model_bars(model_summary):
    plt.figure(figsize=(8, 5))
    plt.bar(model_summary["model_name"], model_summary["accuracy"])
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Overall Accuracy by Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overall_accuracy_by_model.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(model_summary["model_name"], model_summary["hallucination_rate"])
    plt.xlabel("Model")
    plt.ylabel("Hallucination Rate")
    plt.title("Hallucination Rate by Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hallucination_rate_by_model.png")
    plt.close()

    if "refusal_rate" in model_summary.columns:
        plt.figure(figsize=(8, 5))
        plt.bar(model_summary["model_name"], model_summary["refusal_rate"])
        plt.xlabel("Model")
        plt.ylabel("Refusal Rate")
        plt.title("Refusal Rate by Model")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "refusal_rate_by_model.png")
        plt.close()


def save_calibration_plot(df, bins=10):
    df = df.copy()
    df["conf_bin"] = pd.cut(df["confidence"], bins=bins, include_lowest=True)

    bucket_summary = df.groupby("conf_bin", observed=False).agg(
        avg_confidence=("confidence", "mean"),
        accuracy=("is_correct", "mean"),
        count=("is_correct", "size"),
    ).dropna()

    plt.figure(figsize=(7, 5))
    plt.plot(
        bucket_summary["avg_confidence"],
        bucket_summary["accuracy"],
        marker="o",
        label="Observed"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.xlabel("Average Confidence")
    plt.ylabel("Actual Accuracy")
    plt.title("Calibration Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_plot.png")
    plt.close()


def main():
    if not INPUT_PATH.exists():
        print(f"Missing file: {INPUT_PATH}")
        return
    if not MODEL_SUMMARY_PATH.exists():
        print(f"Missing file: {MODEL_SUMMARY_PATH}")
        return
    if not CATEGORY_SUMMARY_PATH.exists():
        print(f"Missing file: {CATEGORY_SUMMARY_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PER_MODEL_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    model_summary = pd.read_csv(MODEL_SUMMARY_PATH)
    category_summary = pd.read_csv(CATEGORY_SUMMARY_PATH)

    df["confidence"] = df["confidence"].astype(float)
    df["is_correct"] = df["is_correct"].astype(int)
    df["hallucination"] = df["hallucination"].astype(int)

    save_heatmap(
        category_summary,
        value_col="accuracy",
        title="Accuracy Heatmap: Model × Category",
        filename="accuracy_heatmap_model_by_category.png",
    )

    save_heatmap(
        category_summary,
        value_col="hallucination_rate",
        title="Hallucination Rate Heatmap: Model × Category",
        filename="hallucination_heatmap_model_by_category.png",
    )

    if "refusal_rate" in category_summary.columns:
        save_heatmap(
            category_summary,
            value_col="refusal_rate",
            title="Refusal Rate Heatmap: Model × Category",
            filename="refusal_heatmap_model_by_category.png",
        )

    save_model_bars(model_summary)
    save_confidence_boxplot(df)
    save_confidence_violin(df)
    save_calibration_plot(df)
    save_per_model_histograms(df)

    print(f"Saved combined plots to: {OUTPUT_DIR}")
    print(f"Saved per-model plots to: {PER_MODEL_DIR}")


if __name__ == "__main__":
    main()