import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "water_level",
    "soil_moisture",
    "water_temp_c",
    "turbidity_ntu",
    "rain_mm",
]


def load_model(model_path: str):
    return joblib.load(model_path)


def build_sample(args):
    # allow psi_kpa to be provided or estimate from soil_moisture_mid
    psi = args.psi_kpa_15cm if getattr(args, "psi_kpa_15cm", None) is not None else -((0.40 - args.soil_moisture_mid) * 100.0)

    sample = {
        "fwl_cm": args.fwl_cm,
        "psi_kpa_15cm": psi,
        "water_level": args.water_level,
        "soil_moisture_top": args.soil_moisture_top,
        "soil_moisture_mid": args.soil_moisture_mid,
        "soil_moisture_deep": args.soil_moisture_deep,
        "groundwater_depth": args.groundwater_depth,
        "water_temp_c": args.water_temp_c,
        "turbidity_ntu": args.turbidity_ntu,
        "rain_mm": args.rain_mm,
        "flow_rate": args.flow_rate,
        "ec_us": args.ec_us,
        "ndvi": args.ndvi,
        "et_mm": args.et_mm,
        "crop_stage": args.crop_stage,
    }
    return pd.DataFrame([sample], columns=FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict gate open/close status for a single hydrological sensor sample."
    )
    parser.add_argument("--model-path", type=str, default="models/gate_model_random_forest.joblib", help="Saved model file.")
    parser.add_argument("--fwl-cm", type=float, required=True, help="Field water level in cm (negative = below surface)")
    parser.add_argument("--psi-kpa-15cm", type=float, required=False, help="Soil matric potential at 15 cm (kPa). If omitted it will be estimated from soil-moisture-mid.")
    parser.add_argument("--water-level", type=float, required=True, help="Water level in meters")
    parser.add_argument("--soil-moisture-top", type=float, required=True, help="Soil moisture top layer (0-1)")
    parser.add_argument("--soil-moisture-mid", type=float, required=True, help="Soil moisture mid layer (0-1)")
    parser.add_argument("--soil-moisture-deep", type=float, required=True, help="Soil moisture deep layer (0-1)")
    parser.add_argument("--groundwater-depth", type=float, required=True, help="Groundwater depth in meters")
    parser.add_argument("--water-temp-c", type=float, required=True)
    parser.add_argument("--turbidity-ntu", type=float, required=True)
    parser.add_argument("--rain-mm", type=float, required=True)
    parser.add_argument("--flow-rate", type=float, required=True)
    parser.add_argument("--ec-us", type=float, required=True)
    parser.add_argument("--ndvi", type=float, required=True)
    parser.add_argument("--et-mm", type=float, required=True)
    parser.add_argument("--crop-stage", type=int, required=True, choices=[0,1,2,3], help="Crop stage: 0=transplanting,1=vegetative,2=reproductive,3=ripening")
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
