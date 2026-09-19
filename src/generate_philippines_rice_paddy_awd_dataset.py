"""Generate a synthetic hourly rice-paddy sensor stream for AWD experiments.

The labels in this dataset are synthetic operational ground truth. They encode
the supplied IRRI AWD and habitat-risk assumptions; they are not field
observations or a validated epidemiological transmission model.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


STAGES = (
    "Transplanting",
    "Vegetative/Tillering",
    "Reproductive/Flowering",
    "Ripening/Maturation",
)
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

MIN_WATER_LEVEL_CM = -20.0
MAX_WATER_LEVEL_CM = 10.0
AWD_DRAWDOWN_CM = -15.0
AWD_REFLOOD_TARGET_CM = 5.0
FLOW_MIN_MS = 0.0
FLOW_MAX_MS = 1.2
PH_MIN = 5.0
PH_MAX = 8.5
SOIL_MOISTURE_MIN = 15.0
SOIL_MOISTURE_MAX = 85.0
SOIL_TEMP_MIN = 14.0
SOIL_TEMP_MAX = 36.0
VEGETATION_COVER_MIN = 10.0
VEGETATION_COVER_MAX = 95.0
VEGETATION_HEIGHT_MIN = 5.0
VEGETATION_HEIGHT_MAX = 120.0


def crop_stage(day_of_year: int) -> str:
    """Return a plausible stage for a 120-day rice crop."""
    cycle_day = day_of_year % 120
    if cycle_day < 14:
        return "Transplanting"
    if cycle_day < 56:
        return "Vegetative/Tillering"
    if cycle_day < 70:
        return "Reproductive/Flowering"
    return "Ripening/Maturation"


def seasonal_rainfall_factor(month: int) -> float:
    """Approximate Philippine wet-season influence, without claiming local observations."""
    return 0.75 if month in {6, 7, 8, 9, 10, 11} else 0.25


def water_table_series(
    timestamps: pd.DatetimeIndex, rng: np.random.Generator
) -> np.ndarray:
    """Create bounded water levels with gradual changes and AWD drawdown cycles."""
    values = np.empty(len(timestamps), dtype=float)
    previous = 2.5
    for index, timestamp in enumerate(timestamps):
        stage = crop_stage(timestamp.dayofyear)
        rainfall = seasonal_rainfall_factor(timestamp.month)
        diurnal = 0.35 * np.sin(2 * np.pi * timestamp.hour / 24.0)
        rain_effect = rainfall * 0.18

        if stage == "Reproductive/Flowering":
            target = 3.5 + rain_effect + diurnal
        else:
            cycle = (timestamp.dayofyear % 18) / 18.0
            target = 4.0 - 20.0 * cycle + rain_effect + diurnal
            if stage == "Transplanting":
                target += 1.5

        innovation = rng.normal(0.0, 0.45)
        previous = 0.86 * previous + 0.14 * target + innovation
        values[index] = np.clip(previous, MIN_WATER_LEVEL_CM, MAX_WATER_LEVEL_CM)
    return values


def calculate_inundation_duration(water_levels: np.ndarray) -> np.ndarray:
    """Count consecutive hourly readings above zero, reported in days."""
    durations = np.zeros(len(water_levels), dtype=float)
    consecutive_hours = 0
    for index, level in enumerate(water_levels):
        if level > 0.0:
            consecutive_hours += 1
        else:
            consecutive_hours = 0
        durations[index] = consecutive_hours / 24.0
    return durations


def generate_dataset(
    start: str = "2024-01-01",
    periods: int = 4320,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a reproducible hourly stream (4320 readings = 180 days)."""
    if not 1000 <= periods <= 5000:
        raise ValueError("periods must be between 1000 and 5000 readings")

    rng = np.random.default_rng(random_seed)
    timestamps = pd.date_range(start=start, periods=periods, freq="h")
    water_level = water_table_series(timestamps, rng)
    flowering_mask = np.array(
        [crop_stage(day) == "Reproductive/Flowering" for day in timestamps.dayofyear]
    )
    water_level[flowering_mask] = np.clip(
        water_level[flowering_mask], 0.01, AWD_REFLOOD_TARGET_CM
    )
    inundation_days = calculate_inundation_duration(water_level)

    stages = np.array([crop_stage(day) for day in timestamps.dayofyear])
    hours = timestamps.hour.to_numpy()
    months = timestamps.month.to_numpy()
    rainfall = np.array([seasonal_rainfall_factor(month) for month in months])

    # Positive standing water raises VWC; drawdown approaches the lower bound.
    moisture = 15.0 + 60.0 * np.clip(
        (water_level - MIN_WATER_LEVEL_CM)
        / (AWD_REFLOOD_TARGET_CM - MIN_WATER_LEVEL_CM),
        0.0,
        1.0,
    )
    moisture += rng.normal(0.0, 2.0, periods) + rainfall * 2.0
    moisture = np.clip(moisture, SOIL_MOISTURE_MIN, SOIL_MOISTURE_MAX)

    solar = np.sin(np.pi * (hours - 6) / 14.0)
    solar = np.clip(solar, 0.0, 1.0)
    soil_temperature = 22.0 + 8.0 * solar + 1.5 * rainfall + rng.normal(0.0, 0.8, periods)
    soil_temperature = np.clip(soil_temperature, SOIL_TEMP_MIN, SOIL_TEMP_MAX)

    flow = 0.10 + 0.025 * np.maximum(water_level, 0.0)
    flow += 0.08 * rainfall + rng.normal(0.0, 0.035, periods)
    flow = np.clip(flow, FLOW_MIN_MS, FLOW_MAX_MS)

    water_ph = 6.7 + 0.15 * rainfall - 0.015 * (soil_temperature - 25.0)
    water_ph += rng.normal(0.0, 0.18, periods)
    water_ph = np.clip(water_ph, PH_MIN, PH_MAX)

    stage_index = {stage: index for index, stage in enumerate(STAGES)}
    stage_progress = np.array(
        [(timestamps[index].dayofyear % 120) / 120.0 for index in range(periods)]
    )
    vegetation_cover = 10.0 + 80.0 * np.minimum(stage_progress * 1.35, 1.0)
    vegetation_cover += rng.normal(0.0, 3.0, periods)
    vegetation_cover = np.clip(vegetation_cover, VEGETATION_COVER_MIN, VEGETATION_COVER_MAX)
    vegetation_height = 5.0 + 115.0 * np.minimum(stage_progress * 1.25, 1.0)
    vegetation_height += rng.normal(0.0, 2.5, periods)
    vegetation_height = np.clip(
        vegetation_height, VEGETATION_HEIGHT_MIN, VEGETATION_HEIGHT_MAX
    )

    high_habitat_conditions = (
        (inundation_days > 14.0)
        & (soil_temperature >= 15.0)
        & (soil_temperature <= 30.0)
        & (water_ph >= 5.5)
        & (water_ph <= 7.9)
        & (moisture >= 20.0)
        & (moisture <= 80.0)
    )
    moderate_conditions = (
        (inundation_days > 3.0)
        & (soil_temperature >= 15.0)
        & (soil_temperature <= 32.0)
        & (water_ph >= 5.3)
        & (water_ph <= 8.1)
        & (moisture >= 20.0)
        & (moisture <= 80.0)
    )
    habitat_risk = np.select(
        [high_habitat_conditions, moderate_conditions],
        ["High", "Moderate"],
        default="Low",
    )

    # Open means reflood/actuate the inlet. Flowering safety takes precedence.
    gate_action = (
        ((stages == "Reproductive/Flowering") & (water_level < 2.0))
        | ((stages != "Reproductive/Flowering") & (water_level <= AWD_DRAWDOWN_CM))
        | ((stages != "Reproductive/Flowering") & (moisture <= 25.0))
    ).astype(int)

    return pd.DataFrame(
        {
            "Month": [MONTH_NAMES[month] for month in months],
            "Day_of_Week": timestamps.day_name(),
            "Hour": hours,
            "Crop_Growth_Stage": stages,
            "Water_Table_Depth_cm": water_level,
            "Inundation_Duration_Days": inundation_days,
            "Water_Flow_Velocity_ms": flow,
            "Water_pH": water_ph,
            "Soil_Moisture_VWC_Percent": moisture,
            "Soil_Temperature_C": soil_temperature,
            "Vegetation_Coverage_Percent": vegetation_cover,
            "Vegetation_Height_cm": vegetation_height,
            "Snail_Habitat_Risk": habitat_risk,
            "AWD_Gate_Action": gate_action,
        }
    )


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise ValueError if physical ranges or label rules are violated."""
    ranges = {
        "Water_Table_Depth_cm": (MIN_WATER_LEVEL_CM, MAX_WATER_LEVEL_CM),
        "Water_Flow_Velocity_ms": (FLOW_MIN_MS, FLOW_MAX_MS),
        "Water_pH": (PH_MIN, PH_MAX),
        "Soil_Moisture_VWC_Percent": (SOIL_MOISTURE_MIN, SOIL_MOISTURE_MAX),
        "Soil_Temperature_C": (SOIL_TEMP_MIN, SOIL_TEMP_MAX),
        "Vegetation_Coverage_Percent": (VEGETATION_COVER_MIN, VEGETATION_COVER_MAX),
        "Vegetation_Height_cm": (VEGETATION_HEIGHT_MIN, VEGETATION_HEIGHT_MAX),
    }
    for column, (lower, upper) in ranges.items():
        if not df[column].between(lower, upper).all():
            raise ValueError(f"{column} violates [{lower}, {upper}]")

    expected_high = (
        (df["Inundation_Duration_Days"] > 14.0)
        & df["Soil_Temperature_C"].between(15.0, 30.0)
        & df["Water_pH"].between(5.5, 7.9)
        & df["Soil_Moisture_VWC_Percent"].between(20.0, 80.0)
    )
    if not (df.loc[expected_high, "Snail_Habitat_Risk"] == "High").all():
        raise ValueError("High-risk rows do not satisfy the specified high-risk rule")
    if (df["Snail_Habitat_Risk"] == "High").loc[~expected_high].any():
        raise ValueError("High-risk label exists outside the specified high-risk rule")

    flowering = df["Crop_Growth_Stage"] == "Reproductive/Flowering"
    if (df.loc[flowering, "Water_Table_Depth_cm"] <= 0.0).any():
        raise ValueError("Flowering rows must remain flooded above 0 cm")
    expected_action = (
        ((flowering) & (df["Water_Table_Depth_cm"] < 2.0))
        | ((~flowering) & (df["Water_Table_Depth_cm"] <= AWD_DRAWDOWN_CM))
        | ((~flowering) & (df["Soil_Moisture_VWC_Percent"] <= 25.0))
    ).astype(int)
    if not (df["AWD_Gate_Action"] == expected_action).all():
        raise ValueError("AWD_Gate_Action does not match the configured rules")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--periods", type=int, default=4320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/philippines_rice_paddy_awd_dataset.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_dataset(args.start, args.periods, args.seed)
    validate_dataset(dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    print(f"Generated {len(dataset):,} rows")
    print(f"Saved dataset to {output}")
    print("\nSummary statistics:")
    print(dataset.describe(include="all").transpose().to_string())
    print("\nLabel distributions:")
    print(dataset["Snail_Habitat_Risk"].value_counts().to_string())
    print(dataset["AWD_Gate_Action"].value_counts().sort_index().to_string())
    print("\nAll physical and logical constraints passed.")


if __name__ == "__main__":
    main()
