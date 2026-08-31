import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "water_level",
    "soil_moisture",
    "temperature",
    "water_velocity",
]
TARGET = "gate_decision"


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict]:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "model": "random_forest",
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, zero_division=0)),
    }

    return model, metrics


def save_outputs(model, metrics: dict, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "gate_model_random_forest.joblib"
    metrics_path = out_dir / "model_metrics_random_forest.json"

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved trained model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest classifier on the synthetic gate-control dataset."
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args.dataset)
    model, metrics = train_model(dataset)
    save_outputs(model, metrics, args.output_dir)
    print("Model:", metrics["model"])
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1_score"])
