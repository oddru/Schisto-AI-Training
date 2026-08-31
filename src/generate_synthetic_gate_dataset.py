import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["water_level", "soil_moisture", "temperature", "water_velocity"]
TARGET = "gate_decision"

WATER_LEVEL_TRIGGER_AWD = -15.0
WATER_LEVEL_TARGET_FLOOD = 5.0
VWC_SATURATED = 0.52
VWC_AWD_TRIGGER = 0.30
VWC_CRITICAL_WILTING = 0.18
MIN_FLUSH_VELOCITY = 0.05


def determine_gate_state(
    water_level: float,
    soil_moisture: float,
    temperature: float,
    water_velocity: float,
    previous_gate_decision: str,
) -> str:
    """Deterministic AWD + schistosomiasis control state machine.

    Agronomic logic:
    - When the gate is closed, the field is kept in a dry-down phase and stays closed until the AWD trigger is reached
      (water_level <= -15 cm or soil_moisture <= 0.30).
    - Once triggered, the gate opens to re-flood the field and remains open until the flood ceiling is reached
      (water_level >= 5 cm and soil_moisture >= 0.52).
    - If the gate is already open but the field is shallow and stagnant (water_level between 0 and 3 cm and
      water_velocity < 0.05 m/s), the controller keeps it open to flush stagnant snail habitats and avoid micro-puddle
      breeding conditions.
    - A critical soil moisture failure mode overrides noisy water-level readings: if the field has dropped below the
      wilting guardrail (soil_moisture <= 0.18) while the water_level sensor falsely appears above the AWD trigger,
      the gate must open to protect the crop.
    """
    previous_state = (previous_gate_decision or "").upper()
    if previous_state not in {"OPEN", "CLOSED"}:
        previous_state = "CLOSED"

    # Sensor-fault/desiccation fallback has highest priority because crop stress mitigation outweighs a noisy water-level reading.
    if soil_moisture <= VWC_CRITICAL_WILTING and water_level > WATER_LEVEL_TRIGGER_AWD:
        return "OPEN"

    if previous_state == "CLOSED":
        if water_level <= WATER_LEVEL_TRIGGER_AWD or soil_moisture <= VWC_AWD_TRIGGER:
            return "OPEN"
        return "CLOSED"

    if previous_state == "OPEN":
        if water_level >= WATER_LEVEL_TARGET_FLOOD and soil_moisture >= VWC_SATURATED:
            return "CLOSED"
        if 0.0 <= water_level <= 3.0 and water_velocity < MIN_FLUSH_VELOCITY and temperature >= 25.0:
            return "OPEN"
        return "OPEN"

    return "CLOSED"


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """Fallback risk score retained for compatibility with the wider evaluation pipeline."""
    score = (
        np.maximum(0.55 - df["water_level"], 0.0) * 2.8
        + np.maximum(0.45 - df["soil_moisture"], 0.0) * 3.6
        + np.maximum(df["temperature"] - 26.0, 0.0) * 0.9
        + np.maximum(0.22 - df["water_velocity"], 0.0) * 4.2
    )
    return score


def generate_dataset(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    df = pd.DataFrame(
        {
            "water_level": rng.uniform(-40.0, 10.0, size=n_samples),
            "soil_moisture": rng.uniform(0.05, 0.60, size=n_samples),
            "temperature": rng.uniform(15.0, 35.0, size=n_samples),
            "water_velocity": rng.uniform(0.0, 0.8, size=n_samples),
        }
    )

    prev_state = "CLOSED"
    gate_states = []
    for _, row in df.iterrows():
        state = determine_gate_state(
            water_level=float(row["water_level"]),
            soil_moisture=float(row["soil_moisture"]),
            temperature=float(row["temperature"]),
            water_velocity=float(row["water_velocity"]),
            previous_gate_decision=prev_state,
        )
        gate_states.append(state)
        prev_state = state

    df["gate_state"] = gate_states
    df[TARGET] = (df["gate_state"] == "OPEN").astype(int)
    df["previous_gate_decision"] = ["CLOSED"] + gate_states[:-1]
    return df[FEATURES + ["gate_state", TARGET, "previous_gate_decision"]]


def save_dataset(df: pd.DataFrame, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dataset with a deterministic AWD + snail-risk gate state machine."
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
    print("Positive rate:", dataset[TARGET].mean())
    print("Columns:", list(dataset.columns))
