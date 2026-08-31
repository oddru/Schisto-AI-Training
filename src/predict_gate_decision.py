import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "water_level",
    "soil_moisture",
    "temperature",
    "water_velocity",
]


def load_model(model_path: str):
    return joblib.load(model_path)


def build_sample(args) -> pd.DataFrame:
    sample = {feature: getattr(args, feature) for feature in FEATURES}
    return pd.DataFrame([sample], columns=FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict gate open/close status for a single sensor sample."
    )
    parser.add_argument("--model-path", type=str, default="models/gate_model_random_forest.joblib", help="Saved model file.")
    parser.add_argument("--water-level", type=float, required=True, help="Water level in meters")
    parser.add_argument("--soil-moisture", type=float, required=True, help="Soil moisture (0-1)")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature in °C")
    parser.add_argument("--water-velocity", type=float, required=True, help="Water velocity in m/s")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model_path)
    sample = build_sample(args)
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0, 1])

    print(json.dumps({
        "prediction": prediction,
        "prediction_label": "gate_open" if prediction == 1 else "gate_close",
        "probability_gate_open": probability,
        "sample": sample.to_dict(orient="records")[0],
    }, indent=2))
