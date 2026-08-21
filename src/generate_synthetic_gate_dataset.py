import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """Domain-informed gate-risk score for stagnant warm water conditions."""
    return (
        np.maximum(0.18 - df["flow_rate"], 0.0) * 2.5
        + np.maximum(df["water_temp_c"] - 24.0, 0.0) * 0.8
        + np.maximum(80.0 - df["turbidity_ntu"], 0.0) * 0.08
        + np.maximum(15.0 - df["rain_mm"], 0.0) * 0.2
    )


def generate_dataset(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """Create a hydrological dataset with a rule-based gate label."""
    rng = np.random.default_rng(random_seed)

    df = pd.DataFrame(
        {
            "flow_rate": rng.uniform(0.03, 0.7, size=n_samples),
            "water_temp_c": rng.uniform(18.0, 34.0, size=n_samples),
            "turbidity_ntu": rng.uniform(15.0, 120.0, size=n_samples),
            "rain_mm": rng.uniform(0.0, 35.0, size=n_samples),
            "conductivity_us": rng.uniform(100.0, 700.0, size=n_samples),
        }
    )

    df["risk_score"] = compute_risk_score(df)
    df["gate_open"] = (df["risk_score"] > 5.0).astype(int)
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
