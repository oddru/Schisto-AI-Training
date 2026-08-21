import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "flow_rate",
    "water_temp_c",
    "turbidity_ntu",
    "rain_mm",
    "conductivity_us",
]


def load_model(model_path: str):
    return joblib.load(model_path)


def build_sample(args):
    sample = {
        "flow_rate": args.flow_rate,
        "water_temp_c": args.water_temp_c,
        "turbidity_ntu": args.turbidity_ntu,
        "rain_mm": args.rain_mm,
        "conductivity_us": args.conductivity_us,
    }
    return pd.DataFrame([sample], columns=FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict gate open/close status for a single hydrological sensor sample."
    )
    parser.add_argument("--model-path", type=str, default="models/gate_model_random_forest.joblib", help="Saved model file.")
    parser.add_argument("--flow-rate", type=float, required=True)
    parser.add_argument("--water-temp-c", type=float, required=True)
    parser.add_argument("--turbidity-ntu", type=float, required=True)
    parser.add_argument("--rain-mm", type=float, required=True)
    parser.add_argument("--conductivity-us", type=float, required=True)
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
