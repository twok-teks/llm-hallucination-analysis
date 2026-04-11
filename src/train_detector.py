import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = Path("reports/merged/all_models_final_results.csv")
METRICS_PATH = Path("reports/metrics/detector_metrics.json")
PLOT_DIR = Path("reports/plots/detector")
PREDICTIONS_DIR = Path("reports/merged")


def save_confusion_matrix(cm, filename):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.xticks([0, 1], ["Not Hallucination", "Hallucination"])
    plt.yticks([0, 1], ["Not Hallucination", "Hallucination"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_feature_importance(feature_names, importances, filename, top_n=20):
    feature_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    plt.barh(feature_df["feature"][::-1], feature_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_roc_curve(y_test, y_scores, filename):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_pr_curve(y_test, y_scores, filename):
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def evaluate_thresholds(y_true, y_scores):
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    rows = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        rows.append({
            "threshold": t,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    return pd.DataFrame(rows)


def main():
    if not INPUT_PATH.exists():
        print(f"Missing file: {INPUT_PATH}")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required_cols = {
        "confidence",
        "question_length",
        "answer_length",
        "category",
        "model_name",
        "question_validity",
        "refusal",
        "hallucination",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns in {INPUT_PATH}: {sorted(missing)}")
        return

    df["confidence"] = df["confidence"].astype(float)
    df["question_length"] = df["question_length"].astype(float)
    df["answer_length"] = df["answer_length"].astype(float)
    df["refusal"] = df["refusal"].astype(int)
    df["hallucination"] = df["hallucination"].astype(int)

    # Engineered features
    df["length_ratio"] = df["answer_length"] / (df["question_length"] + 1.0)
    df["confidence_x_answer_length"] = df["confidence"] * df["answer_length"]
    df["low_confidence_flag"] = (df["confidence"] < 0.5).astype(int)

    feature_cols = [
        "confidence",
        "question_length",
        "answer_length",
        "length_ratio",
        "confidence_x_answer_length",
        "low_confidence_flag",
        "refusal",
        "category",
        "model_name",
        "question_validity",
    ]

    X = df[feature_cols]
    y = df["hallucination"]

    numeric_features = [
        "confidence",
        "question_length",
        "answer_length",
        "length_ratio",
        "confidence_x_answer_length",
        "low_confidence_flag",
        "refusal",
    ]
    categorical_features = [
        "category",
        "model_name",
        "question_validity",
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logreg_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    rf_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced_subsample",
        )),
    ])

    models = {
        "logistic_regression": logreg_model,
        "random_forest": rf_model,
    }

    metrics_output = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_scores = model.predict_proba(X_test)[:, 1]

        # Default threshold for reporting
        threshold = 0.40
        y_pred = (y_scores >= threshold).astype(int)

        metrics_output[model_name] = {
            "threshold_used": threshold,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_scores),
            "average_precision": average_precision_score(y_test, y_scores),
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0
            ),
        }

        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(cm, PLOT_DIR / f"{model_name}_confusion_matrix.png")
        save_roc_curve(y_test, y_scores, PLOT_DIR / f"{model_name}_roc_curve.png")
        save_pr_curve(y_test, y_scores, PLOT_DIR / f"{model_name}_pr_curve.png")

        threshold_df = evaluate_thresholds(y_test, y_scores)
        threshold_df.to_csv(
            PREDICTIONS_DIR / f"{model_name}_thresholds.csv",
            index=False,
        )

        results_df = X_test.copy().reset_index(drop=True)
        results_df["actual"] = y_test.reset_index(drop=True)
        results_df["predicted"] = y_pred
        results_df["predicted_score"] = y_scores
        results_df["detector_model"] = model_name
        results_df.to_csv(
            PREDICTIONS_DIR / f"{model_name}_detector_predictions.csv",
            index=False,
        )

        if model_name == "random_forest":
            preprocessor_fit = model.named_steps["preprocessor"]
            classifier_fit = model.named_steps["classifier"]

            feature_names = preprocessor_fit.get_feature_names_out()
            importances = classifier_fit.feature_importances_
            save_feature_importance(
                feature_names,
                importances,
                PLOT_DIR / "random_forest_feature_importance.png",
            )

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)

    print(f"Saved detector metrics to: {METRICS_PATH}")
    print(f"Saved detector plots to: {PLOT_DIR}")
    print(f"Saved threshold tables and predictions to: {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()