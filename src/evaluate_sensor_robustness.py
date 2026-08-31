import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score

FEATURES = ["water_level", "soil_moisture", "temperature", "water_velocity"]
TARGET = "gate_decision"
NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

np.random.seed(42)


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_model(model_path: str):
    return joblib.load(model_path)


def get_feature_bounds(df: pd.DataFrame) -> dict:
    bounds = {}
    for feature in FEATURES:
        if feature == "soil_moisture":
            bounds[feature] = (0.0, 0.60)
        else:
            bounds[feature] = (float(df[feature].min()), float(df[feature].max()))
    return bounds


def clip_features(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    clipped = df.copy()
    for feature in FEATURES:
        lower, upper = bounds[feature]
        clipped[feature] = np.clip(clipped[feature], lower, upper)
    return clipped


def add_gaussian_noise(df: pd.DataFrame, noise_level: float, bounds: dict) -> pd.DataFrame:
    noisy = df.copy()
    for feature in FEATURES:
        sigma = noise_level * noisy[feature].std(ddof=0)
        noisy[feature] = noisy[feature] + np.random.normal(0.0, sigma, len(noisy))
    return clip_features(noisy, bounds)


def add_impulse_noise(df: pd.DataFrame, noise_level: float, bounds: dict) -> pd.DataFrame:
    noisy = df.copy()
    n_rows = len(noisy)
    n_corrupted = max(1, int(round(n_rows * noise_level)))
    selected_rows = np.random.choice(n_rows, size=n_corrupted, replace=False)

    for feature in FEATURES:
        lower, upper = bounds[feature]
        replacements = np.random.choice([lower, upper], size=n_corrupted)
        noisy.loc[selected_rows, feature] = replacements
    return clip_features(noisy, bounds)


def add_sensor_drift(df: pd.DataFrame, noise_level: float, bounds: dict) -> pd.DataFrame:
    noisy = df.copy()
    for feature in FEATURES:
        lower, upper = bounds[feature]
        drift = noise_level * (upper - lower)
        noisy[feature] = noisy[feature] + drift
    return clip_features(noisy, bounds)


def add_sensor_freeze(df: pd.DataFrame, noise_level: float, bounds: dict) -> pd.DataFrame:
    noisy = df.copy()
    n_rows = len(noisy)
    if n_rows == 0:
        return noisy

    block_count = max(1, int(round(n_rows * noise_level)))
    block_size = max(2, int(np.ceil(n_rows / max(block_count, 1))))

    for _ in range(block_count):
        start = int(np.random.randint(0, max(1, n_rows - block_size + 1)))
        end = min(n_rows, start + block_size)
        for feature in FEATURES:
            frozen_value = noisy.loc[start, feature]
            if start == 0:
                noisy.loc[0:end, feature] = frozen_value
            else:
                noisy.loc[start:end, feature] = frozen_value
    return clip_features(noisy, bounds)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def format_metric(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_summary_table(noise_level: float, rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Fault Type": [row["Fault Type"] for row in rows],
            "Accuracy": [row["Accuracy"] for row in rows],
            "Precision": [row["Precision"] for row in rows],
            "Recall": [row["Recall"] for row in rows],
            "F1-Score": [row["F1-Score"] for row in rows],
            "MAE": [row["MAE"] for row in rows],
        }
    )
    table["Accuracy"] = table["Accuracy"].map(format_metric)
    table["Precision"] = table["Precision"].map(format_metric)
    table["Recall"] = table["Recall"].map(format_metric)
    table["F1-Score"] = table["F1-Score"].map(format_metric)
    table["MAE"] = table["MAE"].map(lambda x: f"{x:.2f}")

    current_header = f"Fault Type ({int(noise_level * 100)}%)"
    table.columns = [current_header, "Accuracy", "Precision", "Recall", "F1-Score", "MAE"]
    return table


def export_results(results: list[dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Random Forest under incremental sensor corruption profiles for the gate-control system."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/synthetic_gate_dataset.csv",
        help="Path to the synthetic labeled dataset CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/gate_model_random_forest.joblib",
        help="Path to the trained model artifact.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sensor_robustness_results.csv",
        help="Path to export the long-format robustness CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)
    model = load_model(args.model_path)

    if set(FEATURES + [TARGET]).difference(df.columns):
        missing = sorted(set(FEATURES + [TARGET]).difference(df.columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X_clean = df[FEATURES].copy()
    y_clean = df[TARGET].copy()
    feature_bounds = get_feature_bounds(df)

    baseline_metrics = evaluate_predictions(y_clean, model.predict(X_clean))
    all_results = []

    for noise_level in NOISE_LEVELS:
        rows = []
        baseline_row = {
            "Fault Type": "Baseline (Clean)",
            "Accuracy": baseline_metrics["accuracy"],
            "Precision": baseline_metrics["precision"],
            "Recall": baseline_metrics["recall"],
            "F1-Score": baseline_metrics["f1_score"],
            "MAE": baseline_metrics["mae"],
        }
        rows.append(baseline_row)
        all_results.append({
            "noise_level": int(noise_level * 100),
            "fault_type": "Baseline (Clean)",
            "accuracy": baseline_metrics["accuracy"],
            "precision": baseline_metrics["precision"],
            "recall": baseline_metrics["recall"],
            "f1_score": baseline_metrics["f1_score"],
            "mae": baseline_metrics["mae"],
        })

        noise_profiles = {
            "Gaussian (EMI)": lambda x: add_gaussian_noise(x, noise_level, feature_bounds),
            "Impulse (Spikes)": lambda x: add_impulse_noise(x, noise_level, feature_bounds),
            "Sensor Drift (Silt)": lambda x: add_sensor_drift(x, noise_level, feature_bounds),
            "Freezing (Stuck)": lambda x: add_sensor_freeze(x, noise_level, feature_bounds),
        }

        for fault_name, noise_fn in noise_profiles.items():
            noisy_df = noise_fn(df[FEATURES].copy())
            y_pred = model.predict(noisy_df)
            metrics = evaluate_predictions(y_clean, y_pred)
            record = {
                "noise_level": int(noise_level * 100),
                "fault_type": fault_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "MAE": metrics["mae"],
            }
            rows.append({
                "Fault Type": fault_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "MAE": metrics["mae"],
            })
            all_results.append({
                "noise_level": int(noise_level * 100),
                "fault_type": fault_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "mae": metrics["mae"],
            })

        summary_table = build_summary_table(noise_level, rows)
        print(summary_table.to_string(index=False))
        print()

    export_results(all_results, args.output)
    print(f"Saved long-format robustness results to {args.output}")


if __name__ == "__main__":
    main()
