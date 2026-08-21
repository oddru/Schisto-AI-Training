import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "flow_rate",
    "water_temp_c",
    "turbidity_ntu",
    "rain_mm",
    "conductivity_us",
]
TARGET = "gate_open"


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame, model_name: str = "random_forest"):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )

    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=4000)
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds)),
        "classification_report": classification_report(
            y_test, preds, output_dict=True, zero_division=0
        ),
    }

    return model, metrics, X_train, X_test, y_train, y_test


def save_outputs(model, metrics: dict, output_dir: str, model_name: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"gate_model_{model_name}.joblib"
    metrics_path = out_dir / f"model_metrics_{model_name}.json"

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved trained model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised model on the synthetic gate-control dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/synthetic_gate_dataset.csv",
        help="Path to the labeled synthetic dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory where the trained model and metrics will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["logistic_regression", "random_forest"],
        help="Which model to train.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args.dataset)
    model, metrics, _, _, _, _ = train_model(dataset, model_name=args.model)
    save_outputs(model, metrics, args.output_dir, args.model)
    print("Model:", metrics["model"])
    print("Accuracy:", metrics["accuracy"])
    print("F1:", metrics["f1_score"])
