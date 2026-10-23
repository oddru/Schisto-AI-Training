"""Run reproducible, chronological Chapter 3 model and safety experiments."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURES = [
    "Month",
    "Day_of_Week",
    "Hour",
    "Crop_Growth_Stage",
    "Water_Table_Depth_cm",
    "Inundation_Duration_Days",
    "Water_Flow_Velocity_ms",
    "Water_pH",
    "Soil_Moisture_VWC_Percent",
    "Soil_Temperature_C",
    "Vegetation_Coverage_Percent",
    "Vegetation_Height_cm",
]
TARGET = "AWD_Gate_Action"
CATEGORICAL_FEATURES = ["Month", "Day_of_Week", "Crop_Growth_Stage"]
NUMERIC_FEATURES = [name for name in FEATURES if name not in CATEGORICAL_FEATURES]
NOISE_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
FAULT_TYPES = ("Gaussian noise", "Impulse spikes", "Sensor drift", "Sensor freezing")
TRAIN_FRACTION = 0.80
RANDOM_STATE = 42

# These are explicit simulation assumptions, not field-calibrated actuator rates.
OPEN_DEPTH_CHANGE_CM_PER_HOUR = 1.0
CLOSED_DEPTH_CHANGE_CM_PER_HOUR = -0.5
AWD_MIN_DEPTH_CM = -15.0
FLOWERING_MIN_DEPTH_CM = 0.0
MAX_DEPTH_CM = 5.0
MAX_READING_AGE_SECONDS = 3600
SENSOR_RANGES = {
    "Hour": (0.0, 23.0),
    "Water_Table_Depth_cm": (-20.0, 10.0),
    "Inundation_Duration_Days": (0.0, 30.0),
    "Water_Flow_Velocity_ms": (0.0, 1.2),
    "Water_pH": (5.0, 8.5),
    "Soil_Moisture_VWC_Percent": (15.0, 85.0),
    "Soil_Temperature_C": (14.0, 36.0),
    "Vegetation_Coverage_Percent": (10.0, 95.0),
    "Vegetation_Height_cm": (5.0, 120.0),
}
SENSOR_SIGNAL_FEATURES = (
    "Water_Table_Depth_cm",
    "Water_Flow_Velocity_ms",
    "Water_pH",
    "Soil_Moisture_VWC_Percent",
    "Soil_Temperature_C",
    "Vegetation_Coverage_Percent",
    "Vegetation_Height_cm",
)


def build_model() -> Pipeline:
    """Create the project Random Forest with explicit, reproducible settings."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def load_and_split(
    dataset_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    dataset = pd.read_csv(dataset_path)
    missing = sorted(set(FEATURES + [TARGET]).difference(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if dataset.isna().any().any():
        raise ValueError("Dataset contains missing values; resolve these before training.")
    if not dataset[TARGET].isin([0, 1]).all():
        raise ValueError("AWD_Gate_Action must contain only binary 0/1 labels.")

    split_at = math.floor(len(dataset) * TRAIN_FRACTION)
    train = dataset.iloc[:split_at].copy()
    test = dataset.iloc[split_at:].copy()
    if train[TARGET].nunique() != 2 or test[TARGET].nunique() != 2:
        raise ValueError("Chronological split must contain both target classes.")

    return train[FEATURES], test[FEATURES], train[TARGET], test[TARGET], dataset


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "mae": float(np.mean(np.abs(y_true.to_numpy() - y_pred))),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "positive_support": int(y_true.sum()),
        "sample_count": int(len(y_true)),
    }


def _bounds(feature: str) -> tuple[float, float]:
    return SENSOR_RANGES[feature]


def corrupt_features(
    features: pd.DataFrame,
    fault_type: str,
    level: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return corrupted features and an oracle mask of rows modified by injection."""
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"Unsupported fault type: {fault_type}")
    if not 0 < level <= 1:
        raise ValueError("Corruption intensity must be in (0, 1].")

    rng = np.random.default_rng(seed)
    corrupted = features.copy()
    fault_mask = np.zeros(len(features), dtype=bool)

    if fault_type == "Gaussian noise":
        for feature in NUMERIC_FEATURES:
            sigma = level * float(features[feature].std(ddof=0))
            corrupted[feature] += rng.normal(0.0, sigma, size=len(features))
        fault_mask[:] = True
    elif fault_type == "Impulse spikes":
        count = max(1, int(round(len(features) * level)))
        indices = rng.choice(len(features), size=count, replace=False)
        fault_mask[indices] = True
        for feature in NUMERIC_FEATURES:
            lower, upper = _bounds(feature)
            corrupted.loc[corrupted.index[indices], feature] = rng.choice(
                [lower, upper], size=count
            )
    elif fault_type == "Sensor drift":
        for feature in NUMERIC_FEATURES:
            lower, upper = _bounds(feature)
            offsets = np.linspace(0.0, level * (upper - lower), len(features))
            corrupted[feature] += offsets
        fault_mask[:] = True
    else:
        count = max(1, int(math.ceil(len(features) * level)))
        start = int(rng.integers(0, len(features) - count + 1))
        stop = start + count
        fault_mask[start:stop] = True
        for feature in NUMERIC_FEATURES:
            frozen_value = corrupted.iloc[start][feature]
            corrupted.iloc[start:stop, corrupted.columns.get_loc(feature)] = frozen_value

    for feature in NUMERIC_FEATURES:
        lower, upper = _bounds(feature)
        corrupted[feature] = corrupted[feature].clip(lower, upper)
    return corrupted, fault_mask


def _lower_water_limit(stage: str) -> float:
    return (
        FLOWERING_MIN_DEPTH_CM
        if stage == "Reproductive/Flowering"
        else AWD_MIN_DEPTH_CM
    )


def one_hour_depth(depth_cm: float, gate_open: bool) -> float:
    """Apply the explicitly assumed one-hour gate response."""
    change = (
        OPEN_DEPTH_CHANGE_CM_PER_HOUR
        if gate_open
        else CLOSED_DEPTH_CHANGE_CM_PER_HOUR
    )
    return depth_cm + change


def violates_water_limits(depth_cm: float, stage: str) -> bool:
    return depth_cm < _lower_water_limit(stage) or depth_cm > MAX_DEPTH_CM


def safety_interlock(requested_open: bool, reading: pd.Series) -> bool:
    """Override a command if its one-hour projection crosses an AWD limit."""
    depth = float(reading["Water_Table_Depth_cm"])
    stage = str(reading["Crop_Growth_Stage"])
    minimum = _lower_water_limit(stage)
    open_projection = one_hour_depth(depth, True)
    close_projection = one_hour_depth(depth, False)
    if open_projection > MAX_DEPTH_CM:
        return False
    if close_projection < minimum:
        return True
    return requested_open


@dataclass
class ControllerResult:
    gate_open: bool
    safe_mode: bool
    reason: str


class SafeGateController:
    """Small software controller with hold-last-state fail-safe semantics."""

    def __init__(self, initial_gate_open: bool = False) -> None:
        self.gate_open = initial_gate_open
        self._last_sensor_values: dict[str, float] = {}
        self._identical_counts = {feature: 0 for feature in SENSOR_SIGNAL_FEATURES}

    def step(
        self,
        reading: pd.Series,
        captured_at: datetime,
        now: datetime,
        fault_detected: bool = False,
        requested_open: bool = False,
    ) -> ControllerResult:
        reason = self._invalid_reason(reading, captured_at, now, fault_detected)
        if not reason and self._is_frozen(reading):
            reason = "frozen_readings"
        if reason:
            return ControllerResult(self.gate_open, True, reason)

        self.gate_open = safety_interlock(requested_open, reading)
        return ControllerResult(self.gate_open, False, "ok")

    def _is_frozen(self, reading: pd.Series) -> bool:
        frozen = False
        for feature in SENSOR_SIGNAL_FEATURES:
            value = float(reading[feature])
            if value == self._last_sensor_values.get(feature):
                self._identical_counts[feature] += 1
            else:
                self._identical_counts[feature] = 0
            self._last_sensor_values[feature] = value
            frozen |= self._identical_counts[feature] >= 2
        return frozen

    @staticmethod
    def _invalid_reason(
        reading: pd.Series,
        captured_at: datetime,
        now: datetime,
        fault_detected: bool,
    ) -> str:
        if fault_detected:
            return "sensor_fault_flag"
        if captured_at.tzinfo is None or now.tzinfo is None:
            return "invalid_timestamp"
        age_seconds = (now - captured_at).total_seconds()
        if age_seconds < 0 or age_seconds > MAX_READING_AGE_SECONDS:
            return "stale_reading"
        for feature in FEATURES:
            if feature not in reading.index:
                return f"missing:{feature}"
        for feature in NUMERIC_FEATURES:
            try:
                value = float(reading[feature])
            except (TypeError, ValueError):
                return f"invalid:{feature}"
            if not math.isfinite(value):
                return f"non_finite:{feature}"
            lower, upper = _bounds(feature)
            if not lower <= value <= upper:
                return f"out_of_range:{feature}"
        if not float(reading["Hour"]).is_integer():
            return "invalid:Hour"
        if reading["Month"] not in {
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        }:
            return "invalid:Month"
        if reading["Day_of_Week"] not in {
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        }:
            return "invalid:Day_of_Week"
        if reading["Crop_Growth_Stage"] not in {
            "Transplanting",
            "Vegetative/Tillering",
            "Reproductive/Flowering",
            "Ripening/Maturation",
        }:
            return "invalid:Crop_Growth_Stage"
        return ""


def evaluate_actuation(
    sensor_readings: pd.DataFrame,
    physical_states: pd.DataFrame,
    predictions: np.ndarray,
    fault_mask: np.ndarray,
) -> dict[str, float | int]:
    """Measure direct and interlocked one-hour outcomes against synthetic states."""
    if not (
        len(sensor_readings)
        == len(physical_states)
        == len(predictions)
        == len(fault_mask)
    ):
        raise ValueError(
            "Sensor readings, physical states, predictions, and fault mask "
            "must have equal lengths."
        )

    direct_violations = 0
    controlled_violations = 0
    safe_modes = 0
    safe_mode_open_commands = 0
    false_negative_actions = 0
    interlock_overrides = 0
    controller = SafeGateController()

    for row_index, ((_, reading), (_, physical)) in enumerate(
        zip(sensor_readings.iterrows(), physical_states.iterrows())
    ):
        actual_depth = float(physical["Water_Table_Depth_cm"])
        stage = str(physical["Crop_Growth_Stage"])
        requested_open = bool(predictions[row_index])

        direct_depth = one_hour_depth(actual_depth, requested_open)
        direct_violations += int(violates_water_limits(direct_depth, stage))
        false_negative_actions += int(
            bool(physical[TARGET]) and not requested_open
        )

        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = controller.step(
            reading,
            timestamp,
            timestamp,
            fault_detected=bool(fault_mask[row_index]),
            requested_open=requested_open,
        )
        if result.safe_mode:
            safe_modes += 1
            safe_mode_open_commands += int(result.gate_open)
        else:
            interlock_overrides += int(result.gate_open != requested_open)
        controlled_depth = one_hour_depth(actual_depth, result.gate_open)
        controlled_violations += int(violates_water_limits(controlled_depth, stage))
    denominator = len(sensor_readings)
    return {
        "sample_count": denominator,
        "direct_boundary_violations": direct_violations,
        "direct_boundary_violation_rate": direct_violations / denominator,
        "controlled_boundary_violations": controlled_violations,
        "controlled_boundary_violation_rate": controlled_violations / denominator,
        "false_negative_gate_actions": false_negative_actions,
        "safe_mode_rows": safe_modes,
        "safe_mode_rate": safe_modes / denominator,
        "safe_mode_open_commands": safe_mode_open_commands,
        "interlock_overrides": interlock_overrides,
    }


def summarize_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = dataset[NUMERIC_FEATURES].agg(["min", "max", "mean", "std"]).T
    numeric.index.name = "feature"
    numeric = numeric.reset_index()
    gate_counts = dataset[TARGET].value_counts().sort_index()
    risk_counts = dataset["Snail_Habitat_Risk"].value_counts().sort_index()
    rows = [
        {
            "label": TARGET,
            "category": str(category),
            "count": int(count),
            "percentage": float(count / len(dataset)),
        }
        for category, count in gate_counts.items()
    ]
    rows.extend(
        {
            "label": "Snail_Habitat_Risk",
            "category": str(category),
            "count": int(count),
            "percentage": float(count / len(dataset)),
        }
        for category, count in risk_counts.items()
    )
    return numeric, pd.DataFrame(rows)


def plot_results(
    labels: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    actuation_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    gate = labels[labels["label"] == TARGET]
    risk = labels[labels["label"] == "Snail_Habitat_Risk"]
    axes[0].bar(gate["category"], gate["count"], color=["#5673b9", "#d9822b"])
    axes[0].set_title("Gate-action label distribution")
    axes[0].set_ylabel("Records")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(risk["category"], risk["count"], color=["#4b9b68", "#e2ad43", "#bc4c4c"])
    axes[1].set_title("Synthetic snail-habitat risk labels")
    axes[1].set_ylabel("Records")
    figure.savefig(output_dir / "Figure_3_1_label_distributions.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    for fault in FAULT_TYPES:
        group = robustness_summary[robustness_summary["fault_type"] == fault]
        axis.plot(
            group["noise_percent"],
            group["f1_score_mean"],
            marker="o",
            label=fault,
        )
        axis.fill_between(
            group["noise_percent"],
            np.maximum(0, group["f1_score_mean"] - group["f1_score_std"]),
            np.minimum(1, group["f1_score_mean"] + group["f1_score_std"]),
            alpha=0.12,
        )
    axis.axhline(
        float(robustness_summary["baseline_f1"].iloc[0]),
        linestyle="--",
        color="black",
        linewidth=1,
        label="Clean baseline",
    )
    axis.set_title("Held-out F1 under injected sensor faults")
    axis.set_xlabel("Injected corruption level (%)")
    axis.set_ylabel("Positive-class F1")
    axis.set_ylim(0, 1.02)
    axis.legend(fontsize=8)
    figure.savefig(output_dir / "Figure_3_2_robustness_curves.png", dpi=200)
    plt.close(figure)

    safety_levels = actuation_summary[
        actuation_summary["fault_type"].eq("Clean baseline")
        | actuation_summary["noise_percent"].isin([5, 15, 30])
    ].copy()
    safety_levels["scenario"] = safety_levels.apply(
        lambda row: (
            "Clean baseline"
            if row["fault_type"] == "Clean baseline"
            else f"{row['fault_type']} ({int(row['noise_percent'])}%)"
        ),
        axis=1,
    )
    y_positions = np.arange(len(safety_levels))
    figure, axis = plt.subplots(
        figsize=(9, 7), constrained_layout=True
    )
    axis.barh(
        y_positions - 0.18,
        safety_levels["direct_boundary_violation_rate_mean"] * 100,
        height=0.35,
        label="Direct model command",
    )
    axis.barh(
        y_positions + 0.18,
        safety_levels["controlled_boundary_violation_rate_mean"] * 100,
        height=0.35,
        label="Interlock + fault-oracle fallback",
    )
    axis.set_yticks(y_positions, safety_levels["scenario"])
    axis.invert_yaxis()
    axis.set_xlabel("Projected one-hour boundary violations (%)")
    axis.set_title("AWD boundary projection under selected conditions")
    axis.legend(fontsize=8)
    figure.savefig(output_dir / "Figure_3_3_awd_boundary_violations.png", dpi=200)
    plt.close(figure)


def run_experiments(
    dataset_path: Path,
    output_dir: Path,
    repeats: int = 10,
) -> dict[str, Path]:
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test, dataset = load_and_split(dataset_path)
    model = build_model()
    model.fit(x_train, y_train)

    baseline_predictions = model.predict(x_test)
    baseline = classification_metrics(y_test, baseline_predictions)
    baseline.update(
        {
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "train_first_row": 0,
            "train_last_row_exclusive": int(len(x_train)),
            "test_first_row": int(len(x_train)),
            "test_last_row_exclusive": int(len(dataset)),
            "split_strategy": "chronological 80/20 holdout",
            "positive_class": 1,
        }
    )

    repeat_rows: list[dict[str, float | int | str]] = []
    actuation_rows: list[dict[str, float | int | str]] = []
    baseline_input = x_test.copy()
    baseline_input[TARGET] = y_test.to_numpy()
    physical_input = x_test.copy()
    physical_input[TARGET] = y_test.to_numpy()
    clean_actuation = evaluate_actuation(
        baseline_input,
        physical_input,
        baseline_predictions,
        np.zeros(len(x_test), dtype=bool),
    )
    actuation_rows.append(
        {"fault_type": "Clean baseline", "noise_percent": 0, **clean_actuation}
    )

    for fault_index, fault_type in enumerate(FAULT_TYPES):
        for level_index, level in enumerate(NOISE_LEVELS):
            for repeat in range(repeats):
                seed = RANDOM_STATE + fault_index * 10_000 + level_index * 100 + repeat
                corrupted, fault_mask = corrupt_features(
                    x_test, fault_type, level, seed
                )
                predictions = model.predict(corrupted)
                metrics = classification_metrics(y_test, predictions)
                repeat_rows.append(
                    {
                        "fault_type": fault_type,
                        "noise_percent": int(level * 100),
                        "repeat": repeat + 1,
                        "seed": seed,
                        **metrics,
                    }
                )

                corrupted_input = corrupted.copy()
                corrupted_input[TARGET] = y_test.to_numpy()
                safety = evaluate_actuation(
                    corrupted_input,
                    physical_input,
                    predictions,
                    fault_mask,
                )
                actuation_rows.append(
                    {
                        "fault_type": fault_type,
                        "noise_percent": int(level * 100),
                        "repeat": repeat + 1,
                        **safety,
                    }
                )

    replicates = pd.DataFrame(repeat_rows)
    metric_names = ["accuracy", "precision", "recall", "f1_score", "mae"]
    summary = (
        replicates.groupby(["fault_type", "noise_percent"], sort=False)[metric_names]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary["repeat_count"] = repeats
    summary["baseline_f1"] = baseline["f1_score"]
    summary["baseline_accuracy"] = baseline["accuracy"]

    dataset_numeric, label_summary = summarize_dataset(dataset)
    actuation_replicates = pd.DataFrame(actuation_rows)
    actuation_metrics = [
        "direct_boundary_violations",
        "direct_boundary_violation_rate",
        "controlled_boundary_violations",
        "controlled_boundary_violation_rate",
        "false_negative_gate_actions",
        "safe_mode_rows",
        "safe_mode_rate",
        "safe_mode_open_commands",
        "interlock_overrides",
    ]
    actuation_summary = (
        actuation_replicates.groupby(["fault_type", "noise_percent"], sort=False)[
            actuation_metrics
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    actuation_summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in actuation_summary.columns
    ]
    dataset_numeric.to_csv(output_dir / "dataset_numeric_summary.csv", index=False)
    label_summary.to_csv(output_dir / "dataset_label_distribution.csv", index=False)
    pd.DataFrame([baseline]).to_csv(output_dir / "baseline_metrics.csv", index=False)
    replicates.to_csv(output_dir / "robustness_replicates.csv", index=False)
    summary.to_csv(output_dir / "robustness_summary.csv", index=False)
    actuation_replicates.to_csv(
        output_dir / "actuation_safety_replicates.csv", index=False
    )
    actuation_summary.to_csv(output_dir / "actuation_safety_summary.csv", index=False)

    with (output_dir / "experiment_assumptions.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": str(dataset_path),
                "dataset_is_synthetic": True,
                "split_strategy": "chronological 80/20 holdout",
                "train_fraction": TRAIN_FRACTION,
                "model": "RandomForestClassifier",
                "n_estimators": 400,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "corruption_levels_percent": [int(level * 100) for level in NOISE_LEVELS],
                "corruption_repeats": repeats,
                "fault_mask_interpretation": (
                    "Oracle injection mask used only to test fallback response; "
                    "not a validated field fault detector."
                ),
                "safe_mode_behavior": "Hold the previous gate command and flag manual review.",
                "open_depth_change_cm_per_hour": OPEN_DEPTH_CHANGE_CM_PER_HOUR,
                "closed_depth_change_cm_per_hour": CLOSED_DEPTH_CHANGE_CM_PER_HOUR,
                "awd_min_depth_cm": AWD_MIN_DEPTH_CM,
                "flowering_min_depth_cm": FLOWERING_MIN_DEPTH_CM,
                "max_depth_cm": MAX_DEPTH_CM,
                "max_reading_age_seconds": MAX_READING_AGE_SECONDS,
            },
            file,
            indent=2,
        )

    plot_results(label_summary, summary, actuation_summary, output_dir)
    return {
        "baseline": output_dir / "baseline_metrics.csv",
        "robustness": output_dir / "robustness_summary.csv",
        "replicates": output_dir / "robustness_replicates.csv",
        "actuation": output_dir / "actuation_safety_summary.csv",
        "actuation_replicates": output_dir / "actuation_safety_replicates.csv",
        "assumptions": output_dir / "experiment_assumptions.json",
        "figure_labels": output_dir / "Figure_3_1_label_distributions.png",
        "figure_results": output_dir / "Figure_3_2_robustness_curves.png",
        "figure_safety": output_dir / "Figure_3_3_awd_boundary_violations.png",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/philippines_rice_paddy_awd_dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/chapter3_experiments"),
    )
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    paths = run_experiments(args.dataset, args.output_dir, args.repeats)
    print(f"Completed Chapter 3 experiments in {time.perf_counter() - started:.1f}s")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
