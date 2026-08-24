import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """Domain-informed AWD irrigation score using multiple sensors.

    Higher score => higher need to irrigate (gate_open = irrigate).

    Components:
    - low water_level increases score (fields allowed to dry under AWD; when low -> need to irrigate)
    - low soil moisture at mid-depth increases score
    - deep groundwater far from surface increases score
    - high ET and NDVI increase crop demand
    - crop_stage adjusts sensitivity (0: transplanting, 1: vegetative, 2: reproductive, 3: ripening)
    """
    # water-level: lower values mean dryer surface (we use 0.5 m reference)
    score = np.maximum(0.35 - df["water_level"], 0.0) * 3.0

    # soil moisture mid-depth is more indicative of plant available water
    score = score + np.maximum(0.30 - df["soil_moisture_mid"], 0.0) * 4.0

    # groundwater depth (meters): larger values mean deeper water table -> more irrigation need
    score = score + np.maximum(df["groundwater_depth"] - 0.15, 0.0) * 2.5

    # crop demand signals
    score = score + np.maximum(df["et_mm"] - 3.0, 0.0) * 0.8
    score = score + np.maximum(df["ndvi"] - 0.35, 0.0) * 2.0

    # temperature and turbidity weakly affect score
    score = score + np.maximum(df["water_temp_c"] - 24.0, 0.0) * 0.3
    score = score + np.maximum(80.0 - df["turbidity_ntu"], 0.0) * 0.04

    # crop stage multiplier: reproductive stages are more sensitive so we increase threshold effect
    stage_mult = np.where(df.get("crop_stage", 1) == 2, 1.2, 1.0)
    return score * stage_mult


def generate_dataset(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """Create a synthetic dataset that supports Safe AWD decision logic.

    Produced columns (numerical):
    - water_level (m)
    - soil_moisture_top, soil_moisture_mid, soil_moisture_deep (volumetric fraction 0-1)
    - groundwater_depth (m)
    - water_temp_c (°C)
    - turbidity_ntu (NTU)
    - rain_mm (mm)
    - flow_rate (m/s) -- canal discharge
    - ec_us (µS/cm)
    - ndvi (0-1)
    - et_mm (mm/day)
    - crop_stage (int: 0=transplanting,1=vegetative,2=reproductive,3=ripening)

    The AWD decision (gate_open) is derived from the generated `awd_score`.
    """
    rng = np.random.default_rng(random_seed)

    df = pd.DataFrame(
        {
            # generate a field water level (FWL) in cm: negative = below surface, positive = standing water
            "fwl_cm": rng.uniform(-40.0, 10.0, size=n_samples),
            "water_level": rng.uniform(0.05, 1.2, size=n_samples),
            "soil_moisture_top": rng.uniform(0.05, 0.6, size=n_samples),
            "soil_moisture_mid": rng.uniform(0.05, 0.55, size=n_samples),
            "soil_moisture_deep": rng.uniform(0.05, 0.5, size=n_samples),
            "groundwater_depth": rng.uniform(0.05, 0.6, size=n_samples),
            "water_temp_c": rng.uniform(18.0, 34.0, size=n_samples),
            "turbidity_ntu": rng.uniform(10.0, 120.0, size=n_samples),
            "rain_mm": rng.uniform(0.0, 35.0, size=n_samples),
            "flow_rate": rng.uniform(0.02, 0.7, size=n_samples),
            "ec_us": rng.uniform(80.0, 1200.0, size=n_samples),
            "ndvi": rng.uniform(0.05, 0.85, size=n_samples),
            "et_mm": rng.uniform(0.5, 6.0, size=n_samples),
            "crop_stage": rng.integers(0, 4, size=n_samples),
        }
    )

    # synthesize a soil matric potential at 15 cm (psi in kPa) from soil_moisture_mid
    # mapping chosen so that soil_moisture_mid ~0.30 -> psi ~ -10 kPa (IRRI threshold)
    df["psi_kpa_15cm"] = -((0.40 - df["soil_moisture_mid"]) * 100.0).clip(-200.0, 0.0)

    df["awd_score"] = compute_risk_score(df)

    # IRRI Safe AWD primary rule: re-irrigate when FWL <= -15 cm or psi <= -10 kPa
    irri_rule = (df["fwl_cm"] <= -15.0) | (df["psi_kpa_15cm"] <= -10.0)

    # gate_open = irrigate when IRRI rule true OR awd_score exceeds threshold
    df["gate_open"] = (irri_rule | (df["awd_score"] > 5.0)).astype(int)
    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic hydrological dataset with a rule-based gate-open/close ground truth."
    )
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of synthetic rows to generate.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_gate_dataset.csv",
        help="Path to save the generated CSV dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = generate_dataset(n_samples=args.n_samples, random_seed=args.random_seed)
    saved = save_dataset(dataset, args.output)
    print(f"Saved synthetic dataset to {saved}")
    print("Positive rate:", dataset["gate_open"].mean())
    print("Columns:", list(dataset.columns))
