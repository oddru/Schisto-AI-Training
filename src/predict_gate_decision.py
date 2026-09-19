import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "Month", "Day_of_Week", "Hour", "Crop_Growth_Stage",
    "Water_Table_Depth_cm", "Inundation_Duration_Days",
    "Water_Flow_Velocity_ms", "Water_pH", "Soil_Moisture_VWC_Percent",
    "Soil_Temperature_C", "Vegetation_Coverage_Percent", "Vegetation_Height_cm",
]


def load_model(model_path: str):
    return joblib.load(model_path)


def build_sample(args) -> pd.DataFrame:
    sample = {
        "Month": args.month,
        "Day_of_Week": args.day_of_week,
        "Hour": args.hour,
        "Crop_Growth_Stage": args.crop_growth_stage,
        "Water_Table_Depth_cm": args.water_table_depth_cm,
        "Inundation_Duration_Days": args.inundation_duration_days,
        "Water_Flow_Velocity_ms": args.water_flow_velocity_ms,
        "Water_pH": args.water_ph,
        "Soil_Moisture_VWC_Percent": args.soil_moisture_vwc_percent,
        "Soil_Temperature_C": args.soil_temperature_c,
        "Vegetation_Coverage_Percent": args.vegetation_coverage_percent,
        "Vegetation_Height_cm": args.vegetation_height_cm,
    }
    return pd.DataFrame([sample], columns=FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict gate open/close status for a single sensor sample."
    )
    parser.add_argument("--model-path", type=str, default="models/gate_model_random_forest.joblib", help="Saved model file.")
    parser.add_argument("--month", required=True)
    parser.add_argument("--day-of-week", required=True)
    parser.add_argument("--hour", type=int, required=True)
    parser.add_argument("--crop-growth-stage", required=True)
    parser.add_argument("--water-table-depth-cm", type=float, required=True)
    parser.add_argument("--inundation-duration-days", type=float, required=True)
    parser.add_argument("--water-flow-velocity-ms", type=float, required=True)
    parser.add_argument("--water-ph", type=float, required=True)
    parser.add_argument("--soil-moisture-vwc-percent", type=float, required=True)
    parser.add_argument("--soil-temperature-c", type=float, required=True)
    parser.add_argument("--vegetation-coverage-percent", type=float, required=True)
    parser.add_argument("--vegetation-height-cm", type=float, required=True)
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
