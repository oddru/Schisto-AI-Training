import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame):
    features = [
        "fwl_cm",
        "psi_kpa_15cm",
        "water_level",
        "soil_moisture_top",
        "soil_moisture_mid",
        "soil_moisture_deep",
        "groundwater_depth",
        "water_temp_c",
        "turbidity_ntu",
        "rain_mm",
        "flow_rate",
        "ec_us",
        "ndvi",
        "et_mm",
        "crop_stage",
    ]
    target = "gate_open"
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )
    model = LogisticRegression(max_iter=4000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Clean evaluation:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))
    return model


def add_gaussian_noise(df: pd.DataFrame, severity: float = 0.05) -> pd.DataFrame:
    noisy = df.copy()
    for col in ["water_level", "soil_moisture_top", "soil_moisture_mid", "soil_moisture_deep", "groundwater_depth", "water_temp_c", "turbidity_ntu", "rain_mm", "flow_rate", "ec_us", "ndvi", "et_mm", "crop_stage"]:
        # for categorical/ordinal (crop_stage) add integer noise; for others add gaussian
        if col == "crop_stage":
            noisy[col] = (noisy[col] + np.random.randint(-1, 2, size=len(noisy))).clip(0, 3)
        else:
            noisy[col] = noisy[col] + np.random.normal(0, severity * noisy[col].std(), len(noisy))
    return noisy


def add_impulsive_noise(df: pd.DataFrame, severity: float = 0.15, p: float = 0.02) -> pd.DataFrame:
    noisy = df.copy()
    idx = np.random.rand(len(noisy)) < p
    for col in ["water_level", "soil_moisture_top", "soil_moisture_mid", "soil_moisture_deep", "groundwater_depth", "water_temp_c", "turbidity_ntu", "rain_mm", "flow_rate", "ec_us", "ndvi", "et_mm"]:
        noise_mag = np.random.choice([-1.0, 1.0], size=idx.sum()) * severity * noisy[col].std()
        noisy.loc[idx, col] = noisy.loc[idx, col] + noise_mag
    # impulsive on crop_stage as occasional misread
    idx2 = np.random.rand(len(noisy)) < p/10
    noisy.loc[idx2, "crop_stage"] = np.random.randint(0, 4, size=idx2.sum())
    return noisy


def add_sensor_drift(df: pd.DataFrame, drift_rate: float = 0.01) -> pd.DataFrame:
    noisy = df.copy()
    for col in ["water_level", "soil_moisture_top", "soil_moisture_mid", "soil_moisture_deep", "groundwater_depth", "water_temp_c", "turbidity_ntu", "rain_mm", "flow_rate", "ec_us", "ndvi", "et_mm"]:
        drift = np.arange(len(noisy)) * drift_rate * noisy[col].std() / max(len(noisy), 1)
        noisy[col] = noisy[col] + drift
    noisy["crop_stage"] = (noisy["crop_stage"] + np.linspace(0, drift_rate*3, len(noisy))).astype(int).clip(0,3)
    return noisy


def add_sensor_freeze(df: pd.DataFrame, freeze_window: int = 150) -> pd.DataFrame:
    noisy = df.copy()
    n_rows = len(noisy)
    start = np.random.randint(0, max(1, n_rows - freeze_window))
    end = min(n_rows, start + freeze_window)
    for col in ["water_level", "soil_moisture_top", "soil_moisture_mid", "soil_moisture_deep", "groundwater_depth", "water_temp_c", "turbidity_ntu", "rain_mm", "flow_rate", "ec_us", "ndvi", "et_mm", "crop_stage"]:
        frozen_value = noisy.loc[start, col]
        noisy.loc[start:end, col] = frozen_value
    return noisy


def evaluate_noise_case(df: pd.DataFrame, noise_name: str, noise_fn, **kwargs):
    noisy = noise_fn(df, **kwargs)
    model = LogisticRegression(max_iter=4000)
    features = [
        "water_level",
        "soil_moisture_top",
        "soil_moisture_mid",
        "soil_moisture_deep",
        "groundwater_depth",
        "water_temp_c",
        "turbidity_ntu",
        "rain_mm",
        "flow_rate",
        "ec_us",
        "ndvi",
        "et_mm",
        "crop_stage",
    ]
    target = "gate_open"

    X_train, X_test, y_train, y_test = train_test_split(
        noisy[features], noisy[target], test_size=0.2, random_state=7, stratify=noisy[target]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"\nNoise case: {noise_name}")
    print("Accuracy:", acc)
    print("F1:", f1)
    print(classification_report(y_test, preds, zero_division=0))
    return {"noise": noise_name, "accuracy": acc, "f1": f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the robustness of a synthetic gate-control model under four sensor-noise scenarios."
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
