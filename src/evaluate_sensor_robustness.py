import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
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


def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET],
    )

    model = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Clean evaluation:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds, zero_division=0))
    print(classification_report(y_test, preds, zero_division=0))
    return model


def add_gaussian_noise(df: pd.DataFrame, severity: float = 0.05) -> pd.DataFrame:
    noisy = df.copy()
    for col in FEATURES:
        noisy[col] = noisy[col] + np.random.normal(0, severity * noisy[col].std(), len(noisy))
    return noisy


def add_impulsive_noise(df: pd.DataFrame, severity: float = 0.15, p: float = 0.02) -> pd.DataFrame:
    noisy = df.copy()
    idx = np.random.rand(len(noisy)) < p
    for col in FEATURES:
        noise_mag = np.random.choice([-1.0, 1.0], size=idx.sum()) * severity * noisy[col].std()
        noisy.loc[idx, col] = noisy.loc[idx, col] + noise_mag
    return noisy


def add_sensor_drift(df: pd.DataFrame, drift_rate: float = 0.01) -> pd.DataFrame:
    noisy = df.copy()
    for col in FEATURES:
        drift = np.arange(len(noisy)) * drift_rate * noisy[col].std() / max(len(noisy), 1)
        noisy[col] = noisy[col] + drift
    return noisy


def add_sensor_freeze(df: pd.DataFrame, freeze_window: int = 150) -> pd.DataFrame:
    noisy = df.copy()
    n_rows = len(noisy)
    start = np.random.randint(0, max(1, n_rows - freeze_window))
    end = min(n_rows, start + freeze_window)
    for col in FEATURES:
        frozen_value = noisy.loc[start, col]
        noisy.loc[start:end, col] = frozen_value
    return noisy


def evaluate_noise_case(df: pd.DataFrame, noise_name: str, noise_fn, **kwargs):
    noisy = noise_fn(df, **kwargs)
    model = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")

    X_train, X_test, y_train, y_test = train_test_split(
        noisy[FEATURES],
        noisy[TARGET],
        test_size=0.2,
        random_state=42,
        stratify=noisy[TARGET],
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"\nNoise case: {noise_name}")
    print("Accuracy:", acc)
    print("F1:", f1)
    print(classification_report(y_test, preds, zero_division=0))
    return {"noise": noise_name, "accuracy": acc, "f1": f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the robustness of a gate-decision Random Forest under sensor-noise scenarios."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/synthetic_gate_dataset.csv",
        help="Path to a synthetic gate-control dataset CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args.dataset)
    print("Loaded dataset with shape:", dataset.shape)
    train_model(dataset)

    evaluate_noise_case(dataset, "gaussian", add_gaussian_noise, severity=0.05)
    evaluate_noise_case(dataset, "impulsive", add_impulsive_noise, severity=0.15, p=0.02)
    evaluate_noise_case(dataset, "drift", add_sensor_drift, drift_rate=0.01)
    evaluate_noise_case(dataset, "freeze", add_sensor_freeze, freeze_window=150)
