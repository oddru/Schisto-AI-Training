# APPENDICES

This document contains the complete first-party code listings associated with the thesis project. It is a source-code companion to the main thesis chapters; the applicable institutional template determines where appendices are placed and how this file is incorporated into the final thesis.

The listings are reproduced from the project files. Generated datasets, downloaded external datasets, model binaries, report outputs, environment/build files, notebook execution outputs, and third-party library implementations are not reproduced here. Notebook entries include code cells only; their Markdown cells and stored outputs are excluded. The repository source files remain the canonical copies.

## Appendix A. Python Application and Experiment Scripts

### A.1 Chapter3 Experiments

Source: `src/chapter3_experiments.py`

```python
"""Run reproducible, chronological Chapter 3 model and safety experiments."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Collection
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

if __package__:
    from .generate_philippines_rice_paddy_awd_dataset import awd_policy_actions
else:
    from generate_philippines_rice_paddy_awd_dataset import awd_policy_actions


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
SENSOR_NUMERIC_FEATURES = [
    name for name in SENSOR_SIGNAL_FEATURES if name in NUMERIC_FEATURES
]


def build_model(class_weight: str | dict[int, float] = "balanced") -> Pipeline:
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
                    class_weight=class_weight,
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


def corrupt_features_by_channel(
    features: pd.DataFrame,
    fault_type: str,
    level: float,
    seed: int,
    affected_features: Collection[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return corrupted inputs and an oracle row-by-channel injection mask."""
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"Unsupported fault type: {fault_type}")
    if not 0 < level <= 1:
        raise ValueError("Corruption intensity must be in (0, 1].")
    selected_features = (
        list(SENSOR_NUMERIC_FEATURES)
        if affected_features is None
        else list(dict.fromkeys(affected_features))
    )
    invalid_features = sorted(set(selected_features) - set(SENSOR_NUMERIC_FEATURES))
    if invalid_features:
        raise ValueError(f"Unsupported sensor feature(s): {invalid_features}")
    if not selected_features:
        raise ValueError("At least one sensor feature must be selected.")
    missing_features = sorted(set(selected_features) - set(features.columns))
    if missing_features:
        raise ValueError(f"Input data is missing sensor feature(s): {missing_features}")

    rng = np.random.default_rng(seed)
    corrupted = features.copy()
    fault_mask = pd.DataFrame(
        False,
        index=features.index,
        columns=SENSOR_NUMERIC_FEATURES,
    )

    if fault_type == "Gaussian noise":
        for feature in selected_features:
            sigma = level * float(features[feature].std(ddof=0))
            corrupted[feature] += rng.normal(0.0, sigma, size=len(features))
            fault_mask[feature] = True
    elif fault_type == "Impulse spikes":
        count = max(1, int(round(len(features) * level)))
        indices = rng.choice(len(features), size=count, replace=False)
        for feature in selected_features:
            lower, upper = _bounds(feature)
            feature_position = corrupted.columns.get_loc(feature)
            mask_position = fault_mask.columns.get_loc(feature)
            corrupted.iloc[indices, feature_position] = rng.choice(
                [lower, upper], size=count
            )
            fault_mask.iloc[indices, mask_position] = True
    elif fault_type == "Sensor drift":
        for feature in selected_features:
            lower, upper = _bounds(feature)
            offsets = np.linspace(0.0, level * (upper - lower), len(features))
            corrupted[feature] += offsets
            fault_mask[feature] = True
    else:
        count = max(1, int(math.ceil(len(features) * level)))
        start = int(rng.integers(0, len(features) - count + 1))
        stop = start + count
        for feature in selected_features:
            frozen_value = corrupted.iloc[start][feature]
            corrupted.iloc[start:stop, corrupted.columns.get_loc(feature)] = frozen_value
            fault_mask.iloc[start:stop, fault_mask.columns.get_loc(feature)] = True

    for feature in selected_features:
        lower, upper = _bounds(feature)
        corrupted[feature] = corrupted[feature].clip(lower, upper)
    return corrupted, fault_mask


def corrupt_features(
    features: pd.DataFrame,
    fault_type: str,
    level: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return corrupted inputs and a row-level oracle mask for compatibility."""
    corrupted, feature_mask = corrupt_features_by_channel(
        features, fault_type, level, seed
    )
    return corrupted, feature_mask.any(axis=1).to_numpy()


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
    """Choose the requested action when safe, otherwise the safer projection."""
    depth = float(reading["Water_Table_Depth_cm"])
    stage = str(reading["Crop_Growth_Stage"])
    minimum = _lower_water_limit(stage)
    open_projection = one_hour_depth(depth, True)
    close_projection = one_hour_depth(depth, False)
    open_is_safe = not violates_water_limits(open_projection, stage)
    close_is_safe = not violates_water_limits(close_projection, stage)
    if requested_open and open_is_safe:
        return True
    if not requested_open and close_is_safe:
        return False
    if open_is_safe:
        return True
    if close_is_safe:
        return False

    open_violation = max(minimum - open_projection, 0.0) + max(
        open_projection - MAX_DEPTH_CM, 0.0
    )
    close_violation = max(minimum - close_projection, 0.0) + max(
        close_projection - MAX_DEPTH_CM, 0.0
    )
    if open_violation == close_violation:
        return requested_open
    return open_violation < close_violation


@dataclass
class ControllerResult:
    gate_open: bool
    safe_mode: bool
    degraded_mode: bool
    reason: str
    policy_requested_open: bool | None
    model_agrees_with_policy: bool | None
    faulted_features: tuple[str, ...]


class SafeGateController:
    """Apply the configured AWD policy; model requests are advisory only."""

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
        faulty_features: Collection[str] = (),
    ) -> ControllerResult:
        unknown_features = sorted(
            set(faulty_features) - set(SENSOR_NUMERIC_FEATURES)
        )
        if unknown_features:
            raise ValueError(f"Unsupported faulted sensor feature(s): {unknown_features}")
        faulted = set(faulty_features)
        reason = self._invalid_reason(reading, captured_at, now, fault_detected)
        frozen_features: set[str] = set()
        if not reason:
            frozen_features = self._frozen_features(reading)
            faulted.update(frozen_features)
        stage = str(reading.get("Crop_Growth_Stage", ""))
        critical_features = {"Water_Table_Depth_cm"}
        if stage != "Reproductive/Flowering":
            critical_features.add("Soil_Moisture_VWC_Percent")
        critical_faults = tuple(sorted(faulted & critical_features))
        if reason:
            return ControllerResult(
                self.gate_open, True, False, reason, None, None, tuple(sorted(faulted))
            )
        if critical_faults:
            return ControllerResult(
                self.gate_open,
                True,
                False,
                f"critical_sensor_fault:{','.join(critical_faults)}",
                None,
                None,
                tuple(sorted(faulted)),
            )

        policy_request = bool(
            awd_policy_actions(
                np.array([reading["Crop_Growth_Stage"]]),
                np.array([reading["Water_Table_Depth_cm"]], dtype=float),
                np.array([reading["Soil_Moisture_VWC_Percent"]], dtype=float),
            )[0]
        )
        degraded = bool(faulted)
        self.gate_open = safety_interlock(policy_request, reading)
        return ControllerResult(
            self.gate_open,
            False,
            degraded,
            f"advisory_sensor_fault:{','.join(sorted(faulted))}" if degraded else "ok",
            policy_request,
            requested_open == policy_request,
            tuple(sorted(faulted)),
        )

    def _frozen_features(self, reading: pd.Series) -> set[str]:
        frozen_features = set()
        for feature in SENSOR_SIGNAL_FEATURES:
            value = float(reading[feature])
            if value == self._last_sensor_values.get(feature):
                self._identical_counts[feature] += 1
            else:
                self._identical_counts[feature] = 0
            self._last_sensor_values[feature] = value
            if self._identical_counts[feature] >= 2:
                frozen_features.add(feature)
        return frozen_features

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
    fault_mask: np.ndarray | pd.DataFrame,
) -> dict[str, float | int]:
    """Compare direct model and policy-controlled projections on synthetic states."""
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
    degraded_modes = 0
    critical_fault_rows = 0
    safe_mode_open_commands = 0
    false_negative_actions = 0
    model_policy_disagreements = 0
    model_policy_comparisons = 0
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
        if isinstance(fault_mask, pd.DataFrame):
            faulty_features = tuple(
                feature
                for feature in SENSOR_NUMERIC_FEATURES
                if bool(fault_mask.iloc[row_index][feature])
            )
            result = controller.step(
                reading,
                timestamp,
                timestamp,
                requested_open=requested_open,
                faulty_features=faulty_features,
            )
        else:
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
            critical_fault_rows += int(
                result.reason.startswith("critical_sensor_fault:")
            )
        else:
            degraded_modes += int(result.degraded_mode)
            interlock_overrides += int(
                result.gate_open != result.policy_requested_open
            )
            model_policy_comparisons += 1
            model_policy_disagreements += int(not result.model_agrees_with_policy)
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
        "critical_fault_rows": critical_fault_rows,
        "degraded_mode_rows": degraded_modes,
        "degraded_mode_rate": degraded_modes / denominator,
        "model_policy_comparisons": model_policy_comparisons,
        "model_policy_disagreements": model_policy_disagreements,
        "model_policy_disagreement_rate": (
            model_policy_disagreements / model_policy_comparisons
            if model_policy_comparisons
            else 0.0
        ),
        "safe_mode_rows": safe_modes,
        "safe_mode_rate": safe_modes / denominator,
        "safe_mode_open_commands": safe_mode_open_commands,
        "interlock_overrides": interlock_overrides,
    }


def evaluate_feature_fault_scopes(
    sensor_readings: pd.DataFrame,
    physical_states: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Measure controller response when one identified sensor channel is faulty."""
    rows = []
    for feature in SENSOR_NUMERIC_FEATURES:
        fault_mask = pd.DataFrame(
            False,
            index=sensor_readings.index,
            columns=SENSOR_NUMERIC_FEATURES,
        )
        fault_mask[feature] = True
        metrics = evaluate_actuation(
            sensor_readings,
            physical_states,
            predictions,
            fault_mask,
        )
        rows.append({"faulted_feature": feature, **metrics})
    return pd.DataFrame(rows)


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
        label="AWD policy + interlock/fault fallback",
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
                corrupted, fault_mask = corrupt_features_by_channel(
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
        "critical_fault_rows",
        "degraded_mode_rows",
        "degraded_mode_rate",
        "model_policy_comparisons",
        "model_policy_disagreements",
        "model_policy_disagreement_rate",
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
    evaluate_feature_fault_scopes(
        baseline_input,
        physical_input,
        baseline_predictions,
    ).to_csv(output_dir / "feature_fault_scope_safety.csv", index=False)

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
                    "Row-by-channel oracle injection mask used only to test "
                    "conditional controller response; not a validated field "
                    "fault detector."
                ),
                "conditional_fault_policy": (
                    "Water depth is critical for every stage. Soil moisture is "
                    "critical outside flowering; faults in other measured channels "
                    "allow the explicit AWD rule to continue in degraded mode."
                ),
                "safe_mode_behavior": "Hold the previous gate command and flag manual review.",
                "controller_authority": (
                    "The configured synthetic AWD policy controls valid readings; "
                    "the Random Forest action is advisory only."
                ),
                "awd_policy": (
                    "During flowering, request open below 2 cm; otherwise request "
                    "open at or below -15 cm or at soil moisture at or below 25%."
                ),
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
        "feature_fault_scope_safety": output_dir / "feature_fault_scope_safety.csv",
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
```

### A.2 Evaluate External Irrigation Data

Source: `src/evaluate_external_irrigation_data.py`

```python
"""Audit and summarize the external irrigation-control time series."""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median


TIMESTAMP_FORMAT = "%d-%b-%Y %H:%M:%S"
DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "external_sources"
    / "Smart irrigation control system data with soil moisture, flow meter and electrovalve relay"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "reports" / "external_irrigation"
)


@dataclass(frozen=True)
class Stream:
    name: str
    value_column: str
    timestamps: list[datetime]
    values: list[float]
    duplicate_timestamps: int


def load_stream(path: Path, value_column: str) -> Stream:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {value_column, "time_stamp"}
        if reader.fieldnames is None or not expected_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns {sorted(expected_columns)}."
            )

        timestamps: list[datetime] = []
        values: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.strptime(row["time_stamp"], TIMESTAMP_FORMAT)
                value = float(row[value_column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {value_column} record at {path}:{row_number}."
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {value_column} at {path}:{row_number}.")
            timestamps.append(timestamp)
            values.append(value)

    if not timestamps:
        raise ValueError(f"{path} contains no data records.")
    if timestamps != sorted(timestamps):
        raise ValueError(f"Timestamps in {path} must be in ascending order.")

    duplicate_count = len(timestamps) - len(set(timestamps))
    return Stream(
        name=path.name,
        value_column=value_column,
        timestamps=timestamps,
        values=values,
        duplicate_timestamps=duplicate_count,
    )


def summarize_stream(
    stream: Stream, expected_interval_seconds: int
) -> dict[str, int | float | str]:
    gaps = [
        (later - earlier).total_seconds()
        for earlier, later in zip(stream.timestamps, stream.timestamps[1:])
    ]
    long_gap_limit = expected_interval_seconds * 1.5
    return {
        "stream": stream.name,
        "value_column": stream.value_column,
        "records": len(stream.values),
        "start": stream.timestamps[0].isoformat(sep=" "),
        "end": stream.timestamps[-1].isoformat(sep=" "),
        "minimum": min(stream.values),
        "median": median(stream.values),
        "maximum": max(stream.values),
        "duplicate_timestamps": stream.duplicate_timestamps,
        "median_gap_seconds": median(gaps) if gaps else 0,
        "gaps_over_1_5x_expected": sum(gap > long_gap_limit for gap in gaps),
        "maximum_gap_seconds": max(gaps, default=0),
    }


def latest_value(
    stream: Stream, timestamp: datetime, max_age_seconds: int
) -> float | None:
    index = bisect_right(stream.timestamps, timestamp) - 1
    if index < 0:
        return None
    age = (timestamp - stream.timestamps[index]).total_seconds()
    if age < 0 or age > max_age_seconds:
        return None
    return stream.values[index]


def summarize_relay_observations(
    relay: Stream, moisture: Stream, moisture_max_age_seconds: int
) -> list[dict[str, int | float]]:
    if any(value not in (0.0, 1.0) for value in relay.values):
        raise ValueError("Relay readings must be binary 0/1 for this audit.")

    paired: dict[int, list[float]] = {0: [], 1: []}
    for timestamp, relay_value in zip(relay.timestamps, relay.values):
        moisture_value = latest_value(moisture, timestamp, moisture_max_age_seconds)
        if moisture_value is not None:
            paired[int(relay_value)].append(moisture_value)

    output: list[dict[str, int | float]] = []
    for state in (0, 1):
        readings = sorted(paired[state])
        output.append(
            {
                "relay_state": state,
                "relay_records": sum(value == state for value in relay.values),
                "paired_moisture_records": len(readings),
                "moisture_min": min(readings) if readings else math.nan,
                "moisture_median": median(readings) if readings else math.nan,
                "moisture_max": max(readings) if readings else math.nan,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty report: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    moisture = load_stream(data_dir / "D_moisture.txt", "moisture")
    flow = load_stream(data_dir / "D_flowmeter.txt", "litres")
    relay = load_stream(data_dir / "D_valve.txt", "relay")

    streams = [
        summarize_stream(moisture, expected_interval_seconds=20 * 60),
        summarize_stream(flow, expected_interval_seconds=5 * 60),
        summarize_stream(relay, expected_interval_seconds=5 * 60),
    ]
    paired = summarize_relay_observations(
        relay, moisture, moisture_max_age_seconds=30 * 60
    )
    transitions = [
        (
            earlier != later,
            (later_time - earlier_time).total_seconds(),
        )
        for earlier, later, earlier_time, later_time in zip(
            relay.values,
            relay.values[1:],
            relay.timestamps,
            relay.timestamps[1:],
        )
    ]
    transition_count = sum(changed for changed, _ in transitions)
    continuous_transition_count = sum(
        changed and gap_seconds <= 10 * 60 for changed, gap_seconds in transitions
    )
    paired_count = sum(row["paired_moisture_records"] for row in paired)
    relay_record_count = len(relay.values)

    output_dir.mkdir(parents=True, exist_ok=True)
    streams_path = output_dir / "stream_quality.csv"
    relay_path = output_dir / "relay_moisture_summary.csv"
    report_path = output_dir / "external_irrigation_data_audit.md"
    write_csv(streams_path, streams)
    write_csv(relay_path, paired)

    relay_counts = Counter(relay.values)
    flow_is_monotonic = all(
        later >= earlier for earlier, later in zip(flow.values, flow.values[1:])
    )
    report_path.write_text(
        "\n".join(
            [
                "# External Irrigation Data Audit",
                "",
                "This report summarizes the three supplied time series without training "
                "or evaluating the project's Random Forest. Source files are read-only; "
                "only aggregate reports are written.",
                "",
                "## Recorded data",
                "",
                f"- Coverage: {min(stream['start'] for stream in streams)} through "
                f"{max(stream['end'] for stream in streams)}.",
                f"- Relay records: {relay_record_count}; state changes between "
                f"adjacent records: {transition_count}, of which "
                f"{continuous_transition_count} occurred across gaps of no more than "
                f"10 minutes; recorded states: "
                f"{dict(sorted(relay_counts.items()))}.",
                f"- Relay records with a preceding moisture reading no more than 30 "
                f"minutes old: {paired_count}/{relay_record_count} "
                f"({paired_count / relay_record_count:.1%}).",
                f"- Flow readings are monotonic in the supplied file: "
                f"{'yes' if flow_is_monotonic else 'no'}. This does not establish "
                "whether the field is a cumulative counter or its unit.",
                "",
                "## Limits on interpretation",
                "",
                "The folder does not provide verified provenance, site/crop context, "
                "a data dictionary, sensor units, relay-state meaning, or confirmation "
                "of the flow-meter semantics. The streams are asynchronous and contain "
                "long gaps; see `stream_quality.csv`. The 20-minute moisture and "
                "5-minute flow/relay reference intervals used to count long gaps are "
                "inferred from the observed cadence, not specified by source metadata. "
                "Moisture summaries by relay state "
                "are descriptive co-observations only, not causal effects, recommended "
                "thresholds, or evidence that logged actions were agronomically safe.",
                "",
                "The records do not contain the current model's required water-table "
                "depth, crop stage, and other features. Therefore they cannot be passed "
                "to the existing classifier or treated as independent gate-action "
                "labels. No model was retrained or changed by this audit.",
                "",
                "## Next evidence needed",
                "",
                "Verify the dataset source/citation, site and crop, channel definitions "
                "and units, whether relay `1` means energized/open, whether `litres` is "
                "a cumulative counter, and whether the data are from a real farm. "
                "Then have an agronomist or system owner confirm which analyses and "
                "safety conclusions are appropriate.",
                "",
                "Detailed aggregate tables: `stream_quality.csv` and "
                "`relay_moisture_summary.csv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report_path, streams_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report_path, streams_path = evaluate(args.data_dir, args.output_dir)
    print(f"External irrigation audit written to {report_path}")
    print(f"Stream quality summary written to {streams_path}")


if __name__ == "__main__":
    main()
```

### A.3 Evaluate Sensor Robustness

Source: `src/evaluate_sensor_robustness.py`

```python
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score

FEATURES = [
    "Month", "Day_of_Week", "Hour", "Crop_Growth_Stage",
    "Water_Table_Depth_cm", "Inundation_Duration_Days",
    "Water_Flow_Velocity_ms", "Water_pH", "Soil_Moisture_VWC_Percent",
    "Soil_Temperature_C", "Vegetation_Coverage_Percent", "Vegetation_Height_cm",
]
NUMERIC_FEATURES = [
    "Hour", "Water_Table_Depth_cm", "Inundation_Duration_Days",
    "Water_Flow_Velocity_ms", "Water_pH", "Soil_Moisture_VWC_Percent",
    "Soil_Temperature_C", "Vegetation_Coverage_Percent", "Vegetation_Height_cm",
]
TARGET = "AWD_Gate_Action"
NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

np.random.seed(42)


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_model(model_path: str):
    return joblib.load(model_path)


def get_feature_bounds(df: pd.DataFrame) -> dict:
    bounds = {}
    for feature in NUMERIC_FEATURES:
        if feature == "Hour":
            bounds[feature] = (0.0, 23.0)
        elif feature == "Water_Table_Depth_cm":
            bounds[feature] = (-20.0, 10.0)
        elif feature == "Water_Flow_Velocity_ms":
            bounds[feature] = (0.0, 1.2)
        elif feature == "Water_pH":
            bounds[feature] = (5.0, 8.5)
        elif feature == "Soil_Moisture_VWC_Percent":
            bounds[feature] = (15.0, 85.0)
        elif feature == "Soil_Temperature_C":
            bounds[feature] = (14.0, 36.0)
        elif feature == "Vegetation_Coverage_Percent":
            bounds[feature] = (10.0, 95.0)
        elif feature == "Vegetation_Height_cm":
            bounds[feature] = (5.0, 120.0)
        else:
            bounds[feature] = (float(df[feature].min()), float(df[feature].max()))
    return bounds


def clip_features(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    clipped = df.copy()
    for feature in NUMERIC_FEATURES:
        lower, upper = bounds[feature]
        clipped[feature] = np.clip(clipped[feature], lower, upper)
    return clipped


def add_gaussian_noise(df: pd.DataFrame, noise_level: float, bounds: dict) -> pd.DataFrame:
    noisy = df.copy()
    for feature in NUMERIC_FEATURES:
        sigma = noise_level * noisy[feature].std(ddof=0)
        noisy[feature] = noisy[feature] + np.random.normal(0.0, sigma, len(noisy))
    return clip_features(noisy, bounds)


def add_impulse_noise(df: pd.DataFrame, noise_level: float, bounds: dict) -> pd.DataFrame:
    noisy = df.copy()
    n_rows = len(noisy)
    n_corrupted = max(1, int(round(n_rows * noise_level)))
    selected_rows = np.random.choice(n_rows, size=n_corrupted, replace=False)

    for feature in NUMERIC_FEATURES:
        lower, upper = bounds[feature]
        replacements = np.random.choice([lower, upper], size=n_corrupted)
        noisy.loc[selected_rows, feature] = replacements
    return clip_features(noisy, bounds)


def add_sensor_drift(df: pd.DataFrame, noise_level: float, bounds: dict) -> pd.DataFrame:
    noisy = df.copy()
    for feature in NUMERIC_FEATURES:
        lower, upper = bounds[feature]
        drift = noise_level * (upper - lower)
        noisy[feature] = noisy[feature] + drift
    return clip_features(noisy, bounds)


def add_sensor_freeze(df: pd.DataFrame, noise_level: float, bounds: dict) -> pd.DataFrame:
    noisy = df.copy()
    n_rows = len(noisy)
    if n_rows == 0:
        return noisy

    block_count = max(1, int(round(n_rows * noise_level)))
    block_size = max(2, int(np.ceil(n_rows / max(block_count, 1))))

    for _ in range(block_count):
        start = int(np.random.randint(0, max(1, n_rows - block_size + 1)))
        end = min(n_rows, start + block_size)
        for feature in NUMERIC_FEATURES:
            frozen_value = noisy.loc[start, feature]
            if start == 0:
                noisy.loc[0:end, feature] = frozen_value
            else:
                noisy.loc[start:end, feature] = frozen_value
    return clip_features(noisy, bounds)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def format_metric(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_summary_table(noise_level: float, rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Fault Type": [row["Fault Type"] for row in rows],
            "Accuracy": [row["Accuracy"] for row in rows],
            "Precision": [row["Precision"] for row in rows],
            "Recall": [row["Recall"] for row in rows],
            "F1-Score": [row["F1-Score"] for row in rows],
            "MAE": [row["MAE"] for row in rows],
        }
    )
    table["Accuracy"] = table["Accuracy"].map(format_metric)
    table["Precision"] = table["Precision"].map(format_metric)
    table["Recall"] = table["Recall"].map(format_metric)
    table["F1-Score"] = table["F1-Score"].map(format_metric)
    table["MAE"] = table["MAE"].map(lambda x: f"{x:.2f}")

    current_header = f"Fault Type ({int(noise_level * 100)}%)"
    table.columns = [current_header, "Accuracy", "Precision", "Recall", "F1-Score", "MAE"]
    return table


def export_results(results: list[dict], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Random Forest under incremental sensor corruption profiles for the gate-control system."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/philippines_rice_paddy_awd_dataset.csv",
        help="Path to the expanded labeled dataset CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/gate_model_random_forest.joblib",
        help="Path to the trained model artifact.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sensor_robustness_results.csv",
        help="Path to export the long-format robustness CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)
    model = load_model(args.model_path)

    if set(FEATURES + [TARGET]).difference(df.columns):
        missing = sorted(set(FEATURES + [TARGET]).difference(df.columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X_clean = df[FEATURES].copy()
    y_clean = df[TARGET].copy()
    feature_bounds = get_feature_bounds(df)

    baseline_metrics = evaluate_predictions(y_clean, model.predict(X_clean))
    all_results = []

    for noise_level in NOISE_LEVELS:
        rows = []
        baseline_row = {
            "Fault Type": "Baseline (Clean)",
            "Accuracy": baseline_metrics["accuracy"],
            "Precision": baseline_metrics["precision"],
            "Recall": baseline_metrics["recall"],
            "F1-Score": baseline_metrics["f1_score"],
            "MAE": baseline_metrics["mae"],
        }
        rows.append(baseline_row)
        all_results.append({
            "noise_level": int(noise_level * 100),
            "fault_type": "Baseline (Clean)",
            "accuracy": baseline_metrics["accuracy"],
            "precision": baseline_metrics["precision"],
            "recall": baseline_metrics["recall"],
            "f1_score": baseline_metrics["f1_score"],
            "mae": baseline_metrics["mae"],
        })

        noise_profiles = {
            "Gaussian (EMI)": lambda x: add_gaussian_noise(x, noise_level, feature_bounds),
            "Impulse (Spikes)": lambda x: add_impulse_noise(x, noise_level, feature_bounds),
            "Sensor Drift (Silt)": lambda x: add_sensor_drift(x, noise_level, feature_bounds),
            "Freezing (Stuck)": lambda x: add_sensor_freeze(x, noise_level, feature_bounds),
        }

        for fault_name, noise_fn in noise_profiles.items():
            noisy_df = noise_fn(df[FEATURES].copy())
            y_pred = model.predict(noisy_df)
            metrics = evaluate_predictions(y_clean, y_pred)
            record = {
                "noise_level": int(noise_level * 100),
                "fault_type": fault_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "MAE": metrics["mae"],
            }
            rows.append({
                "Fault Type": fault_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "MAE": metrics["mae"],
            })
            all_results.append({
                "noise_level": int(noise_level * 100),
                "fault_type": fault_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "mae": metrics["mae"],
            })

        summary_table = build_summary_table(noise_level, rows)
        print(summary_table.to_string(index=False))
        print()

    export_results(all_results, args.output)
    print(f"Saved long-format robustness results to {args.output}")


if __name__ == "__main__":
    main()
```

### A.4 Generate Chapter3 Figures

Source: `src/generate_chapter3_figures.py`

```python
"""Generate supplementary Chapter 3 figures from saved experiment summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "chapter3_tuning"
DEFAULT_AUDIT_DIR = ROOT / "reports" / "external_irrigation"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def add_box(ax, center, text, color, width=0.2, height=0.16, fontsize=9):
    x, y = center
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=color,
        edgecolor="#334155",
        linewidth=1,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def add_arrow(ax, start, end, color="#475569", style="-|>", linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=1.3,
            color=color,
            linestyle=linestyle,
            connectionstyle="arc3,rad=0",
        )
    )


def plot_evaluation_architecture(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Evaluation design and authority boundaries",
        fontsize=15,
        weight="bold",
        pad=16,
    )

    synthetic = "#dbeafe"
    controller = "#dcfce7"
    advisory = "#fef3c7"
    external = "#f1f5f9"
    safe = "#fee2e2"

    add_box(ax, (0.12, 0.78), "Synthetic data\nand rule-derived labels", synthetic, 0.2, 0.16)
    add_box(ax, (0.38, 0.78), "Model fitting\nand validation selection", synthetic, 0.2, 0.16)
    add_box(ax, (0.66, 0.78), "Locked synthetic test\nand fault stress tests", synthetic, 0.22, 0.16)
    add_arrow(ax, (0.22, 0.78), (0.27, 0.78))
    add_arrow(ax, (0.48, 0.78), (0.54, 0.78))

    ax.text(0.5, 0.61, "Python controller evaluation", ha="center", fontsize=11, weight="bold")
    add_box(ax, (0.12, 0.45), "Input validation\nand channel fault flags", controller, 0.2, 0.16)
    add_box(ax, (0.4, 0.45), "Explicit AWD rule\n(valid readings)", controller, 0.2, 0.16)
    add_box(ax, (0.69, 0.45), "One-hour boundary\ninterlock", controller, 0.2, 0.16)
    add_box(ax, (0.91, 0.45), "Projected gate\nresponse", controller, 0.15, 0.16)
    add_arrow(ax, (0.22, 0.45), (0.3, 0.45))
    add_arrow(ax, (0.5, 0.45), (0.59, 0.45))
    add_arrow(ax, (0.79, 0.45), (0.83, 0.45))

    add_box(ax, (0.4, 0.24), "Random Forest output\nadvisory comparison only", advisory, 0.23, 0.15)
    add_box(ax, (0.65, 0.24), "Model-policy agreement\nand prediction metrics", external, 0.2, 0.15)
    add_arrow(ax, (0.66, 0.69), (0.47, 0.32), color="#a16207", linestyle="--")
    add_arrow(ax, (0.515, 0.24), (0.55, 0.24), color="#a16207", linestyle="--")
    add_arrow(ax, (0.69, 0.37), (0.67, 0.32), color="#a16207", linestyle="--")

    add_box(ax, (0.14, 0.24), "Critical / invalid input:\nsafe mode; hold last command", safe, 0.23, 0.15)
    add_arrow(ax, (0.12, 0.36), (0.14, 0.33), color="#b91c1c")

    add_box(ax, (0.89, 0.18), "External sensor/relay logs\nread-only exploratory audit", "#f1f5f9", 0.18, 0.15)
    ax.text(
        0.89,
        0.055,
        "Not used for model training or validation:\nfeature and provenance details are unverified",
        ha="center",
        va="center",
        fontsize=8,
        color="#475569",
    )
    ax.text(
        0.5,
        0.02,
        "Solid arrows: implemented control/evaluation flow. Dashed arrows: advisory comparison only.",
        ha="center",
        fontsize=8,
        color="#475569",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_external_audit(audit_dir: Path, output_path: Path) -> None:
    streams = read_rows(audit_dir / "stream_quality.csv")
    relay_moisture = read_rows(audit_dir / "relay_moisture_summary.csv")
    names = ["Moisture", "Flow meter", "Relay"]
    row_by_name = {row["stream"]: row for row in streams}
    files = ["D_moisture.txt", "D_flowmeter.txt", "D_valve.txt"]
    colors = ["#2563eb", "#0f766e", "#7c3aed"]
    record_counts = [int(row_by_name[file]["records"]) for file in files]
    expected_minutes = [20, 5, 5]
    median_minutes = [
        float(row_by_name[file]["median_gap_seconds"]) / 60 for file in files
    ]
    long_gap_counts = [
        int(row_by_name[file]["gaps_over_1_5x_expected"]) for file in files
    ]
    max_gap_hours = [
        float(row_by_name[file]["maximum_gap_seconds"]) / 3600 for file in files
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Exploratory profile of user-supplied irrigation logs",
        fontsize=15,
        weight="bold",
    )

    axes[0, 0].bar(names, record_counts, color=colors)
    axes[0, 0].set_title("Recorded samples")
    axes[0, 0].set_ylabel("Rows")
    axes[0, 0].bar_label(axes[0, 0].containers[0], fmt="%d", padding=3, fontsize=8)

    x = np.arange(len(names))
    width = 0.36
    axes[0, 1].bar(
        x - width / 2,
        expected_minutes,
        width,
        label="Inferred reference interval",
        color="#cbd5e1",
    )
    axes[0, 1].bar(
        x + width / 2, median_minutes, width, label="Median observed interval", color=colors
    )
    axes[0, 1].set_xticks(x, names)
    axes[0, 1].set_ylabel("Minutes")
    axes[0, 1].set_title("Typical sampling interval")
    axes[0, 1].legend(frameon=False, fontsize=8)

    bars = axes[1, 0].bar(names, long_gap_counts, color=colors)
    axes[1, 0].set_title("Gaps > 1.5× inferred reference interval")
    axes[1, 0].set_ylabel("Gap count")
    axes[1, 0].bar_label(bars, fmt="%d", padding=3, fontsize=8)
    for index, hours in enumerate(max_gap_hours):
        axes[1, 0].text(
            index,
            long_gap_counts[index] * 0.53,
            f"max {hours:.1f} h",
            ha="center",
            color="white",
            fontsize=8,
            weight="bold",
        )

    for row, color in zip(relay_moisture, ["#64748b", "#dc2626"]):
        state = int(row["relay_state"])
        minimum = float(row["moisture_min"])
        center = float(row["moisture_median"])
        maximum = float(row["moisture_max"])
        axes[1, 1].errorbar(
            state,
            center,
            yerr=[[center - minimum], [maximum - center]],
            fmt="o",
            capsize=5,
            markersize=7,
            color=color,
            label=f"Relay state {state}",
        )
    axes[1, 1].set_xticks([0, 1], ["0", "1"])
    axes[1, 1].set_xlabel("Recorded relay value (meaning unverified)")
    axes[1, 1].set_ylabel("Recorded moisture value (unit unverified)")
    axes[1, 1].set_title("Moisture observations by recorded relay value")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.01,
        "Reference intervals inferred from observed cadence, not source metadata. Descriptive "
        "audit only; long gaps and unknown channel semantics limit interpretation; "
        "these logs are not model training or validation data.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_evaluation_architecture(
        args.output_dir / "Figure_3_7_evaluation_architecture.png"
    )
    plot_external_audit(args.audit_dir, args.output_dir / "Figure_3_8_external_data_audit.png")
    print(f"Supplementary Chapter 3 figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
```

### A.5 Generate Philippines Rice Paddy Awd Dataset

Source: `src/generate_philippines_rice_paddy_awd_dataset.py`

```python
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


def awd_policy_actions(
    crop_stages: np.ndarray,
    water_levels: np.ndarray,
    soil_moisture: np.ndarray,
) -> np.ndarray:
    """Apply the configured synthetic gate-action policy to aligned inputs."""
    crop_stages = np.asarray(crop_stages)
    water_levels = np.asarray(water_levels, dtype=float)
    soil_moisture = np.asarray(soil_moisture, dtype=float)
    if not (
        crop_stages.shape == water_levels.shape == soil_moisture.shape
    ):
        raise ValueError("Crop stages, water levels, and moisture must have matching shapes.")

    flowering = crop_stages == "Reproductive/Flowering"
    return (
        (flowering & (water_levels < 2.0))
        | (~flowering & ((water_levels <= AWD_DRAWDOWN_CM) | (soil_moisture <= 25.0)))
    )


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
    gate_action = awd_policy_actions(stages, water_level, moisture).astype(int)

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
```

### A.6 Generate Robustness Report

Source: `src/generate_robustness_report.py`

```python
"""Render the expanded-dataset robustness CSV as a simple HTML report."""

from pathlib import Path

import pandas as pd


def render(input_path: str, output_path: str) -> None:
    results = pd.read_csv(input_path)
    sections = []
    for noise_level, group in results.groupby("noise_level", sort=True):
        rows = []
        for _, row in group.iterrows():
            rows.append(
                "<tr>"
                f"<td>{row['fault_type']}</td>"
                f"<td>{row['accuracy']:.2%}</td>"
                f"<td>{row['precision']:.2%}</td>"
                f"<td>{row['recall']:.2%}</td>"
                f"<td>{row['f1_score']:.2%}</td>"
                f"<td>{row['mae']:.4f}</td>"
                "</tr>"
            )
        sections.append(
            f"<h2>Fault Type ({int(noise_level)}%)</h2>"
            "<table><thead><tr>"
            f"<th>Fault Type ({int(noise_level)}%)</th><th>Accuracy</th>"
            "<th>Precision</th><th>Recall</th><th>F1-Score</th><th>MAE</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>AWD Gate Robustness Evaluation</title>
<style>
body{background:#000;color:#bafecf;font-family:"Courier New",monospace;padding:20px}
main{max-width:1100px;margin:auto;border:1px solid #2efc7f;padding:18px;background:#06110c}
h1,h2{color:#7efc9d;text-transform:uppercase;letter-spacing:.08em}
table{width:100%;border-collapse:collapse;margin:0 0 24px}
th,td{border:1px solid #2efc7f;padding:8px;text-align:center}
th{background:#123f26;color:#d7ffe7} td{background:#0c2214}
</style></head><body><main>
<h1>AWD Gate Robustness Evaluation</h1>
<p>Random Forest trained on the expanded synthetic Philippine rice-paddy dataset.
Metrics are computed against the synthetic AWD_Gate_Action labels.</p>
""" + "".join(sections) + "</main></body></html>"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    render(
        "data/sensor_robustness_results.csv",
        "reports/model_evaluation_report.html",
    )
    print("Wrote reports/model_evaluation_report.html")
```

### A.7 Predict Gate Decision

Source: `src/predict_gate_decision.py`

```python
import argparse
import json

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


def predict_sample(
    model,
    sample: pd.DataFrame,
    threshold: float | None = None,
) -> tuple[int, float]:
    """Return an advisory model prediction and its positive-class probability."""
    probability = float(model.predict_proba(sample)[0, 1])
    if threshold is None:
        return int(model.predict(sample)[0]), probability
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Probability threshold must be between 0 and 1.")
    return int(probability >= threshold), probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an advisory gate-action prediction for one sensor sample; "
            "this is not a field control command."
        )
    )
    parser.add_argument("--model-path", type=str, default="models/gate_model_random_forest.joblib", help="Saved model file.")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional positive-class probability threshold; omit to use the model default.",
    )
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
    prediction, probability = predict_sample(model, sample, args.threshold)

    print(json.dumps({
        "prediction": prediction,
        "prediction_label": "gate_open" if prediction == 1 else "gate_close",
        "prediction_role": "advisory_only",
        "probability_gate_open": probability,
        "sample": sample.to_dict(orient="records")[0],
    }, indent=2))
```

### A.8 Train Gate Model

Source: `src/train_gate_model.py`

```python
﻿import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict]:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "model": "random_forest",
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, zero_division=0)),
    }

    return model, metrics


def save_outputs(model, metrics: dict, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "gate_model_random_forest.joblib"
    metrics_path = out_dir / "model_metrics_random_forest.json"

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved trained model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest classifier on the expanded Philippine rice-paddy dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/philippines_rice_paddy_awd_dataset.csv",
        help="Path to the expanded labeled dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory where the trained model and metrics will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args.dataset)
    model, metrics = train_model(dataset)
    save_outputs(model, metrics, args.output_dir)
    print("Model:", metrics["model"])
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1_score"])
```

### A.9 Tune Gate Model

Source: `src/tune_gate_model.py`

```python
"""Tune a recall-prioritized gate model on independent synthetic realizations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

from src.chapter3_experiments import (
    FEATURES,
    FAULT_TYPES,
    NOISE_LEVELS,
    RANDOM_STATE,
    TARGET,
    build_model,
    classification_metrics,
    corrupt_features_by_channel,
    evaluate_actuation,
    evaluate_feature_fault_scopes,
    summarize_dataset,
)
from src.generate_philippines_rice_paddy_awd_dataset import (
    generate_dataset,
    validate_dataset,
)


TRAIN_SEEDS = (42, 43, 44, 45)
VALIDATION_SEED = 46
TEST_SEED = 47
CLASS_WEIGHT_CANDIDATES: tuple[tuple[str, str | dict[int, float]], ...] = (
    ("balanced", "balanced"),
    ("positive_8x", {0: 1.0, 1: 8.0}),
    ("positive_16x", {0: 1.0, 1: 16.0}),
    ("positive_32x", {0: 1.0, 1: 32.0}),
)
THRESHOLDS = tuple(round(float(value), 2) for value in np.arange(0.05, 0.951, 0.01))


def make_scenario(seed: int) -> pd.DataFrame:
    """Generate one independent RNG realization under the same assumptions."""
    scenario = generate_dataset(start="2024-01-01", periods=4320, random_seed=seed)
    validate_dataset(scenario)
    return scenario


def positive_f2(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))


def tune_threshold(
    probabilities: np.ndarray,
    y_validation: pd.Series,
    class_weight_name: str,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    rows: list[dict[str, float | int | str]] = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        metrics = classification_metrics(y_validation, predictions)
        rows.append(
            {
                "class_weight": class_weight_name,
                "threshold": threshold,
                "f2_score": positive_f2(y_validation, predictions),
                **metrics,
            }
        )
    curve = pd.DataFrame(rows)
    best = curve.sort_values(
        ["f2_score", "precision", "threshold"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0]
    return curve, {
        "class_weight": str(best["class_weight"]),
        "threshold": float(best["threshold"]),
        "f2_score": float(best["f2_score"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "accuracy": float(best["accuracy"]),
        "f1_score": float(best["f1_score"]),
    }


def evaluate_test_predictions(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    class_weight_name: str,
) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = classification_metrics(y_test, predictions)
    metrics.update(
        {
            "class_weight": class_weight_name,
            "threshold": threshold,
            "f2_score": positive_f2(y_test, predictions),
        }
    )
    return metrics, probabilities, predictions


def run_tuning_experiment(
    output_dir: Path,
    repeats: int = 10,
) -> dict[str, Path]:
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_frames = [make_scenario(seed) for seed in TRAIN_SEEDS]
    training_data = pd.concat(training_frames, ignore_index=True)
    validation_data = make_scenario(VALIDATION_SEED)
    test_data = make_scenario(TEST_SEED)

    x_train = training_data[FEATURES]
    y_train = training_data[TARGET]
    x_validation = validation_data[FEATURES]
    y_validation = validation_data[TARGET]
    x_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    tuning_curves: list[pd.DataFrame] = []
    selected_candidates: list[dict[str, float | str]] = []
    fitted_models = {}
    validation_probabilities_by_candidate: dict[str, np.ndarray] = {}

    for candidate_name, class_weight in CLASS_WEIGHT_CANDIDATES:
        model = build_model(class_weight=class_weight)
        model.fit(x_train, y_train)
        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        curve, selected = tune_threshold(
            validation_probabilities,
            y_validation,
            candidate_name,
        )
        tuning_curves.append(curve)
        selected_candidates.append(selected)
        fitted_models[candidate_name] = model
        validation_probabilities_by_candidate[candidate_name] = validation_probabilities

    candidate_table = pd.DataFrame(selected_candidates).sort_values(
        ["f2_score", "precision", "threshold"],
        ascending=[False, False, False],
        kind="stable",
    )
    chosen = candidate_table.iloc[0].to_dict()
    chosen_name = str(chosen["class_weight"])
    chosen_threshold = float(chosen["threshold"])
    tuned_model = fitted_models[chosen_name]

    # Final metrics are computed only after model/threshold selection is locked.
    tuned_test, tuned_probabilities, tuned_predictions = evaluate_test_predictions(
        tuned_model,
        x_test,
        y_test,
        chosen_threshold,
        chosen_name,
    )
    balanced_model = fitted_models["balanced"]
    baseline_test, baseline_probabilities, baseline_predictions = (
        evaluate_test_predictions(
            balanced_model,
            x_test,
            y_test,
            0.5,
            "balanced",
        )
    )

    tuning_curve = pd.concat(tuning_curves, ignore_index=True)
    candidate_table.to_csv(output_dir / "validation_model_selection.csv", index=False)
    tuning_curve.to_csv(output_dir / "validation_threshold_curves.csv", index=False)
    validation_baseline_probabilities = validation_probabilities_by_candidate["balanced"]
    validation_baseline = classification_metrics(
        y_validation,
        (validation_baseline_probabilities >= 0.5).astype(int),
    )
    validation_baseline.update(
        {
            "evaluation": "balanced_default_threshold",
            "class_weight": "balanced",
            "threshold": 0.5,
            "f2_score": positive_f2(
                y_validation,
                (validation_baseline_probabilities >= 0.5).astype(int),
            ),
        }
    )
    selected_validation_row = candidate_table.iloc[0]
    selected_validation = {
        "evaluation": "validation_selected",
        **selected_validation_row.to_dict(),
    }
    pd.DataFrame([validation_baseline, selected_validation]).to_csv(
        output_dir / "validation_operating_point_comparison.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"evaluation": "balanced_default_threshold", **baseline_test},
            {"evaluation": "tuned_model_on_locked_test", **tuned_test},
        ]
    ).to_csv(output_dir / "locked_test_model_comparison.csv", index=False)

    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "balanced_probability": baseline_probabilities,
            "balanced_prediction": baseline_predictions,
            "tuned_probability": tuned_probabilities,
            "tuned_prediction": tuned_predictions,
        }
    ).to_csv(output_dir / "locked_test_predictions.csv", index=False)

    joblib.dump(tuned_model, output_dir / "gate_model_tuned.joblib")

    scenario_label_rows = []
    for seed, scenario in [
        *((seed, frame) for seed, frame in zip(TRAIN_SEEDS, training_frames)),
        (VALIDATION_SEED, validation_data),
        (TEST_SEED, test_data),
    ]:
        _, label_summary = summarize_dataset(scenario)
        label_summary.insert(0, "seed", seed)
        scenario_label_rows.append(label_summary)
    pd.concat(scenario_label_rows, ignore_index=True).to_csv(
        output_dir / "scenario_label_distributions.csv",
        index=False,
    )
    baseline_safety = evaluate_actuation(
        x_test.assign(**{TARGET: y_test.to_numpy()}),
        test_data[FEATURES + [TARGET]],
        baseline_predictions,
        np.zeros(len(test_data), dtype=bool),
    )
    tuned_safety = evaluate_actuation(
        x_test.assign(**{TARGET: y_test.to_numpy()}),
        test_data[FEATURES + [TARGET]],
        tuned_predictions,
        np.zeros(len(test_data), dtype=bool),
    )
    clean_safety_rows = [
        {"evaluation": "balanced_default_threshold", **baseline_safety},
        {"evaluation": "tuned_model_on_locked_test", **tuned_safety},
    ]

    robustness_rows: list[dict[str, float | int | str]] = []
    actuation_rows: list[dict[str, float | int | str]] = [
        {
            "fault_type": "Clean baseline",
            "noise_percent": 0,
            "repeat": 0,
            **tuned_safety,
        }
    ]
    for fault_index, fault_type in enumerate(FAULT_TYPES):
        for level_index, level in enumerate(NOISE_LEVELS):
            for repeat in range(repeats):
                seed = (
                    RANDOM_STATE
                    + 100_000
                    + fault_index * 10_000
                    + level_index * 100
                    + repeat
                )
                corrupted, fault_mask = corrupt_features_by_channel(
                    x_test,
                    fault_type,
                    level,
                    seed,
                )
                metrics, _, predictions = evaluate_test_predictions(
                    tuned_model,
                    corrupted,
                    y_test,
                    chosen_threshold,
                    chosen_name,
                )
                robustness_rows.append(
                    {
                        "fault_type": fault_type,
                        "noise_percent": int(level * 100),
                        "repeat": repeat + 1,
                        "seed": seed,
                        **metrics,
                    }
                )
                safety = evaluate_actuation(
                    corrupted.assign(**{TARGET: y_test.to_numpy()}),
                    test_data[FEATURES + [TARGET]],
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

    robustness_replicates = pd.DataFrame(robustness_rows)
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "mae",
        "f2_score",
    ]
    robustness_summary = (
        robustness_replicates.groupby(["fault_type", "noise_percent"], sort=False)[
            metric_names
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    robustness_summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in robustness_summary.columns
    ]
    robustness_summary["repeat_count"] = repeats
    robustness_summary["locked_threshold"] = chosen_threshold
    robustness_replicates.to_csv(
        output_dir / "robustness_replicates.csv",
        index=False,
    )
    robustness_summary.to_csv(output_dir / "robustness_summary.csv", index=False)

    actuation_replicates = pd.DataFrame(actuation_rows)
    actuation_metrics = [
        "direct_boundary_violations",
        "direct_boundary_violation_rate",
        "controlled_boundary_violations",
        "controlled_boundary_violation_rate",
        "false_negative_gate_actions",
        "critical_fault_rows",
        "degraded_mode_rows",
        "degraded_mode_rate",
        "model_policy_comparisons",
        "model_policy_disagreements",
        "model_policy_disagreement_rate",
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
    actuation_replicates.to_csv(
        output_dir / "actuation_safety_replicates.csv",
        index=False,
    )
    actuation_summary.to_csv(
        output_dir / "actuation_safety_summary.csv",
        index=False,
    )
    evaluate_feature_fault_scopes(
        x_test.assign(**{TARGET: y_test.to_numpy()}),
        test_data[FEATURES + [TARGET]],
        tuned_predictions,
    ).to_csv(output_dir / "feature_fault_scope_safety.csv", index=False)
    pd.DataFrame(clean_safety_rows).to_csv(
        output_dir / "clean_baseline_safety_comparison.csv",
        index=False,
    )

    assumptions = {
        "dataset_is_synthetic": True,
        "training_seeds": list(TRAIN_SEEDS),
        "validation_seed": VALIDATION_SEED,
        "locked_test_seed": TEST_SEED,
        "scenario_rows": 4320,
        "training_rows": int(len(training_data)),
        "validation_rows": int(len(validation_data)),
        "locked_test_rows": int(len(test_data)),
        "class_weight_candidates": [name for name, _ in CLASS_WEIGHT_CANDIDATES],
        "threshold_grid": list(THRESHOLDS),
        "threshold_selection_metric": "positive-class F2 (beta=2)",
        "locked_test_used_for_tuning": False,
        "model": "400-tree RandomForestClassifier",
        "controller_authority": (
            "The configured synthetic AWD policy controls valid readings; "
            "the Random Forest action is advisory only."
        ),
        "awd_policy": (
            "During flowering, request open below 2 cm; otherwise request open "
            "at or below -15 cm or at soil moisture at or below 25%."
        ),
        "random_state": RANDOM_STATE,
        "same_generator_assumptions_across_seeds": True,
        "robustness_repeats": repeats,
        "fallback_fault_mask": (
            "Row-by-channel oracle injection mask; evaluates conditional controller "
            "response and is not a validated field fault detector."
        ),
        "conditional_fault_policy": (
            "Water depth is critical for every stage. Soil moisture is critical "
            "outside flowering; other measured-channel faults allow the explicit "
            "AWD rule to continue in degraded mode."
        ),
        "generalization_scope": (
            "Unseen random realizations from the same synthetic generator; "
            "not field-data validation."
        ),
        "selected_class_weight": chosen_name,
        "selected_threshold": chosen_threshold,
    }
    (output_dir / "tuning_assumptions.json").write_text(
        json.dumps(assumptions, indent=2),
        encoding="utf-8",
    )

    test_comparison = pd.DataFrame(
        [
            {"evaluation": "balanced_default_threshold", **baseline_test},
            {"evaluation": "tuned_model_on_locked_test", **tuned_test},
        ]
    )
    plot_tuning_results(tuning_curve, candidate_table, test_comparison, output_dir)
    plot_robustness_and_safety(
        robustness_summary,
        actuation_summary,
        test_comparison,
        output_dir,
    )
    return {
        "selection": output_dir / "validation_model_selection.csv",
        "validation_comparison": output_dir / "validation_operating_point_comparison.csv",
        "thresholds": output_dir / "validation_threshold_curves.csv",
        "test_comparison": output_dir / "locked_test_model_comparison.csv",
        "robustness": output_dir / "robustness_summary.csv",
        "safety": output_dir / "actuation_safety_summary.csv",
        "feature_fault_scope_safety": output_dir / "feature_fault_scope_safety.csv",
        "assumptions": output_dir / "tuning_assumptions.json",
        "model": output_dir / "gate_model_tuned.joblib",
        "figures": output_dir / "Figure_3_2_validation_thresholds.png",
        "test_figure": output_dir / "Figure_3_3_locked_test_comparison.png",
        "robustness_figure": output_dir / "Figure_3_4_robustness_curves.png",
        "safety_figure": output_dir / "Figure_3_5_awd_safety.png",
        "precision_recall_figure": output_dir / "Figure_3_6_fault_precision_recall.png",
    }


def plot_tuning_results(
    threshold_curve: pd.DataFrame,
    candidates: pd.DataFrame,
    test_comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    for name, group in threshold_curve.groupby("class_weight", sort=False):
        axis.plot(
            group["threshold"],
            group["recall"],
            label=f"{name} recall",
            alpha=0.8,
        )
        axis.plot(
            group["threshold"],
            group["precision"],
            linestyle=":",
            label=f"{name} precision",
            alpha=0.8,
        )
    selected = candidates.iloc[0]
    axis.axvline(
        float(selected["threshold"]),
        color="black",
        linestyle="--",
        label=f"Selected threshold ({selected['threshold']:.2f})",
    )
    axis.set(
        title="Validation precision-recall trade-off by class weight",
        xlabel="Probability threshold",
        ylabel="Validation metric",
        ylim=(0, 1.02),
    )
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(output_dir / "Figure_3_2_validation_thresholds.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    names = test_comparison["evaluation"].tolist()
    x_positions = np.arange(len(names))
    axes[0].bar(x_positions, test_comparison["recall"] * 100)
    axes[0].set_xticks(x_positions, ["Balanced\n0.50 threshold", "Validation-\nselected"])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Positive-class recall (%)")
    axes[0].set_title("Locked synthetic test recall")
    axes[1].bar(x_positions - 0.18, test_comparison["precision"] * 100, width=0.36, label="Precision")
    axes[1].bar(x_positions + 0.18, test_comparison["recall"] * 100, width=0.36, label="Recall")
    axes[1].set_xticks(x_positions, ["Balanced\n0.50 threshold", "Validation-\nselected"])
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Percent")
    axes[1].set_title("Precision-recall trade-off")
    axes[1].legend()
    figure.savefig(output_dir / "Figure_3_3_locked_test_comparison.png", dpi=200)
    plt.close(figure)


def plot_robustness_and_safety(
    robustness: pd.DataFrame,
    safety: pd.DataFrame,
    test_comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean_f1 = float(
        test_comparison.loc[
            test_comparison["evaluation"] == "tuned_model_on_locked_test",
            "f1_score",
        ].iloc[0]
    )
    figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    axis.axhline(clean_f1, color="black", linestyle="--", label="Clean test F1")
    for fault_type, group in robustness.groupby("fault_type", sort=False):
        group = group.sort_values("noise_percent")
        axis.plot(
            group["noise_percent"],
            group["f1_score_mean"],
            marker="o",
            label=fault_type,
        )
        axis.fill_between(
            group["noise_percent"],
            np.maximum(0, group["f1_score_mean"] - group["f1_score_std"].fillna(0)),
            np.minimum(1, group["f1_score_mean"] + group["f1_score_std"].fillna(0)),
            alpha=0.12,
        )
    axis.set(
        title="Locked-test F1 under simulated sensor faults",
        xlabel="Corruption intensity (%)",
        ylabel="Positive-class F1-score",
        ylim=(0, 1.02),
    )
    axis.legend(fontsize=8)
    figure.savefig(output_dir / "Figure_3_4_robustness_curves.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    clean = safety[safety["fault_type"] == "Clean baseline"].iloc[0]
    for fault_type in FAULT_TYPES:
        group = safety[safety["fault_type"] == fault_type].sort_values("noise_percent")
        axes[0].plot(
            group["noise_percent"],
            group["direct_boundary_violation_rate_mean"] * 100,
            marker="o",
            label=fault_type,
        )
        axes[1].plot(
            group["noise_percent"],
            group["controlled_boundary_violation_rate_mean"] * 100,
            marker="o",
            label=fault_type,
        )
    for axis, field in zip(
        axes,
        (
            "direct_boundary_violation_rate_mean",
            "controlled_boundary_violation_rate_mean",
        ),
    ):
        axis.axhline(
            float(clean[field]) * 100,
            color="black",
            linestyle="--",
            label="Clean baseline",
        )
        axis.set_xlabel("Corruption intensity (%)")
        axis.set_ylabel("Projected boundary violations (%)")
    axes[0].set_title("Direct model actuation")
    axes[1].set_title("AWD policy, interlock, and fallback")
    axes[1].legend(fontsize=7)
    figure.savefig(output_dir / "Figure_3_5_awd_safety.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, fault_type in zip(axes.flat, FAULT_TYPES):
        group = robustness[robustness["fault_type"] == fault_type].sort_values(
            "noise_percent"
        )
        intensity = group["noise_percent"].to_numpy()
        for metric, color in (("precision", "#3569a8"), ("recall", "#d17a22")):
            mean = group[f"{metric}_mean"].to_numpy() * 100
            std = group[f"{metric}_std"].fillna(0).to_numpy() * 100
            axis.plot(intensity, mean, marker="o", label=metric.title(), color=color)
            axis.fill_between(
                intensity,
                np.maximum(0, mean - std),
                np.minimum(100, mean + std),
                color=color,
                alpha=0.15,
            )
        axis.axhline(100, color="#3569a8", linestyle=":", linewidth=0.9)
        axis.axhline(
            float(test_comparison.loc[
                test_comparison["evaluation"] == "tuned_model_on_locked_test",
                "recall",
            ].iloc[0]) * 100,
            color="#d17a22",
            linestyle=":",
            linewidth=0.9,
        )
        axis.set_title(fault_type)
        axis.set_xlabel("Corruption intensity (%)")
        axis.set_ylabel("Positive-class metric (%)")
        axis.set_xticks([5, 10, 15, 20, 25, 30])
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(loc="best", fontsize=8)
    figure.suptitle(
        "Precision-recall response to TFCST faults (mean ± SD, 10 trials)",
        fontsize=12,
    )
    figure.savefig(output_dir / "Figure_3_6_fault_precision_recall.png", dpi=200)
    plt.close(figure)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/chapter3_tuning"),
    )
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    outputs = run_tuning_experiment(args.output_dir, args.repeats)
    print(f"Completed tuning and locked evaluation in {time.perf_counter() - started:.1f}s")
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
```

## Appendix B. Isolated External Forecasting Scripts

### B.1 Download Data

Source: `experiments/irrigation_forecasting/download_data.py`

```python
"""Download only the soil-moisture Parquet files needed for this experiment."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


REVISION = "1580f0627902a29bf563f31f7174c3f59ee1713b"
FILES = {
    "field_sensor_data_consortium0.parquet": (
        "7dbdd398cdeda6d62bd39446d91d62398be44c5b3ec4619b28c1e0e305ed9e2a"
    ),
    "field_sensor_data_consortium2.parquet": (
        "ba4a76219b6d746769e6120920f343f36e28e8778f2b88f85d401929f5ef3484"
    ),
}
BASE_URL = (
    "https://huggingface.co/datasets/it4lia/soil_moisture_dataset/resolve/"
    f"{REVISION}"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_data(data_dir: Path) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename, expected_hash in FILES.items():
        target = data_dir / filename
        if target.exists():
            actual_hash = sha256_file(target)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Existing file failed SHA-256 verification: {target}"
                )
        else:
            request = urllib.request.Request(
                f"{BASE_URL}/{filename}",
                headers={"User-Agent": "Schisto-AI-Training-experiment/1.0"},
            )
            temporary = target.with_suffix(target.suffix + ".part")
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    with temporary.open("wb") as handle:
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                actual_hash = sha256_file(temporary)
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"Downloaded file failed SHA-256 verification: {filename}"
                    )
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        downloaded.append(target)
        print(f"Verified {target}")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    download_data(args.data_dir)


if __name__ == "__main__":
    main()
```

### B.2 Evaluate

Source: `experiments/irrigation_forecasting/evaluate.py`

```python
"""Evaluate one-day-ahead external soil-moisture forecasts, separately."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


FEATURE_COLUMNS = [
    "lag_1_day",
    "lag_2_days",
    "lag_7_days",
    "day_of_year_sin",
    "day_of_year_cos",
]
REQUIRED_COLUMNS = {
    "result_time",
    "datastream_name",
    "result",
    "sensor_type",
}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RANDOM_STATE = 42


def load_water_content(data_dir: Path) -> pd.DataFrame:
    paths = sorted(data_dir.glob("field_sensor_data_consortium*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No field_sensor_data_consortium*.parquet files found in {data_dir}. "
            "Run the experiment's download_data module first."
        )

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        if "result_time" not in frame.columns and frame.index.name == "result_time":
            frame = frame.reset_index()
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        consortium = path.stem.rsplit("consortium", 1)[-1]
        selected = frame[list(REQUIRED_COLUMNS)].copy()
        selected["consortium"] = f"consortium{consortium}"
        frames.append(selected)

    sensors = pd.concat(frames, ignore_index=True)
    sensors = sensors.loc[sensors["sensor_type"] == "Water Content"].copy()
    if sensors.empty:
        raise ValueError("No Water Content sensor records were found.")

    sensors["result_time"] = pd.to_datetime(sensors["result_time"], errors="raise")
    sensors["result"] = pd.to_numeric(sensors["result"], errors="coerce")
    sensors = sensors.dropna(subset=["datastream_name", "result_time", "result"])
    if not np.isfinite(sensors["result"].to_numpy(dtype=float)).all():
        raise ValueError("Water Content readings contain infinite values.")
    if sensors.duplicated(["datastream_name", "result_time"]).any():
        raise ValueError(
            "Duplicate timestamps exist within a Water Content stream; "
            "resolve them before forecasting."
        )
    return sensors.sort_values(["datastream_name", "result_time"]).reset_index(drop=True)


def build_forecasting_rows(sensors: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (consortium, stream_name), stream in sensors.groupby(
        ["consortium", "datastream_name"], sort=True
    ):
        readings = stream.set_index("result_time")["result"].sort_index()
        if readings.index.has_duplicates:
            raise ValueError(f"Duplicate timestamps in stream {stream_name!r}.")

        for target_time, target_value in readings.items():
            lag_times = {
                days: target_time - pd.Timedelta(days=days) for days in (1, 2, 7)
            }
            if not all(lag_time in readings.index for lag_time in lag_times.values()):
                continue
            day_of_year = target_time.day_of_year
            angle = 2 * math.pi * day_of_year / 366.0
            records.append(
                {
                    "consortium": consortium,
                    "datastream_name": stream_name,
                    "target_time": target_time,
                    "target": float(target_value),
                    "lag_1_day": float(readings.loc[lag_times[1]]),
                    "lag_2_days": float(readings.loc[lag_times[2]]),
                    "lag_7_days": float(readings.loc[lag_times[7]]),
                    "day_of_year_sin": math.sin(angle),
                    "day_of_year_cos": math.cos(angle),
                }
            )

    rows = pd.DataFrame.from_records(records)
    if rows.empty:
        raise ValueError(
            "No samples have observations at exactly 1, 2, and 7 days before "
            "the target. The dataset may be too sparse for this experiment."
        )
    return rows.sort_values(["target_time", "datastream_name"]).reset_index(drop=True)


def split_by_time(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.Index(rows["target_time"].drop_duplicates().sort_values())
    if len(dates) < 10:
        raise ValueError("At least ten distinct target dates are required.")
    train_end = max(1, math.floor(0.60 * len(dates)))
    validation_end = max(train_end + 1, math.floor(0.80 * len(dates)))
    if validation_end >= len(dates):
        raise ValueError("Not enough distinct dates for chronological test split.")

    train_dates = dates[:train_end]
    validation_dates = dates[train_end:validation_end]
    test_dates = dates[validation_end:]
    train = rows.loc[rows["target_time"].isin(train_dates)].copy()
    validation = rows.loc[rows["target_time"].isin(validation_dates)].copy()
    test = rows.loc[rows["target_time"].isin(test_dates)].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition.")
    return train, validation, test


def score_predictions(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)) if len(target) >= 2 else math.nan,
    }


def evaluate(data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    sensors = load_water_content(data_dir)
    rows = build_forecasting_rows(sensors)
    predictions_by_consortium: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    split_summary: dict[str, dict[str, object]] = {}
    results: list[dict[str, int | float | str]] = []

    for consortium, cohort_rows in rows.groupby("consortium", sort=True):
        train, validation, test = split_by_time(cohort_rows)
        fit = pd.concat([train, validation], ignore_index=True)
        model = XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=500,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(fit[FEATURE_COLUMNS], fit["target"])

        target = test["target"].to_numpy(dtype=float)
        predictions = {
            "XGBoost": model.predict(test[FEATURE_COLUMNS]),
            "Persistence (1 day)": test["lag_1_day"].to_numpy(dtype=float),
            "Seven-day naive": test["lag_7_days"].to_numpy(dtype=float),
        }
        predictions_by_consortium[consortium] = {
            name: (target, prediction) for name, prediction in predictions.items()
        }
        split_summary[consortium] = {
            "sensor_streams": int(cohort_rows["datastream_name"].nunique()),
            "usable_samples": int(len(cohort_rows)),
            "training_samples": int(len(train)),
            "validation_samples": int(len(validation)),
            "test_samples": int(len(test)),
            "train_target_dates": [
                train["target_time"].min().isoformat(),
                train["target_time"].max().isoformat(),
            ],
            "validation_target_dates": [
                validation["target_time"].min().isoformat(),
                validation["target_time"].max().isoformat(),
            ],
            "test_target_dates": [
                test["target_time"].min().isoformat(),
                test["target_time"].max().isoformat(),
            ],
        }
        for name, prediction in predictions.items():
            results.append(
                {
                    "cohort": consortium,
                    "model": name,
                    "test_samples": len(target),
                    **score_predictions(target, prediction),
                }
            )

    for name in ("XGBoost", "Persistence (1 day)", "Seven-day naive"):
        all_targets = np.concatenate(
            [scores[name][0] for scores in predictions_by_consortium.values()]
        )
        all_predictions = np.concatenate(
            [scores[name][1] for scores in predictions_by_consortium.values()]
        )
        results.append(
            {
                "cohort": "Pooled test records",
                "model": name,
                "test_samples": len(all_targets),
                "mae": float(mean_absolute_error(all_targets, all_predictions)),
                "rmse": float(np.sqrt(mean_squared_error(all_targets, all_predictions))),
                "r2": math.nan,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "forecast_metrics.csv"
    pd.DataFrame(results).to_csv(metrics_path, index=False)
    figure_path = output_dir / "forecast_metrics.png"
    pooled_results = pd.DataFrame(
        [row for row in results if row["cohort"] == "Pooled test records"]
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, label in zip(
        axes, ("mae", "rmse"), ("MAE", "RMSE")
    ):
        axis.bar(pooled_results["model"], pooled_results[metric], color=["#2563eb", "#64748b", "#94a3b8"])
        axis.set_title(f"Pooled chronological test {label}")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle("External soil-moisture forecast comparison")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary_path = output_dir / "experiment_summary.json"
    summary = {
        "dataset": "it4lia/soil_moisture_dataset",
        "dataset_provider": "Fondazione Bruno Kessler (FBK) OpenIoT research unit",
        "dataset_region": "Trentino, Italy",
        "dataset_license": "CC BY-NC-ND 4.0",
        "dataset_card_url": (
            "https://huggingface.co/datasets/it4lia/soil_moisture_dataset"
        ),
        "method_reference": (
            "https://huggingface.co/it4lia/predictive_irrigation_models"
        ),
        "task": "one-day-ahead Water Content sensor forecasting",
        "sensor_streams": int(sensors["datastream_name"].nunique()),
        "water_content_records": int(len(sensors)),
        "usable_daily_samples": int(len(rows)),
        "chronological_splits_by_consortium": split_summary,
        "test_samples_pooled": int(
            sum(details["test_samples"] for details in split_summary.values())
        ),
        "features": FEATURE_COLUMNS,
        "model_parameters": {
            "objective": "reg:absoluteerror",
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        },
        "limitations": [
            "Forecasting results do not validate AWD gate-action classification.",
            "Only exact daily lag observations are included; irregular intervals are excluded.",
            "Geographic coverage is Trentino, Italy and does not establish transfer to Philippine rice paddies.",
            "Pooled R-squared is omitted because between-consortium level differences can make it misleading.",
            "Dataset provenance and feature descriptions follow the Hugging Face dataset card.",
            "Do not redistribute the dataset, model artifacts, or derived results without reviewing the CC BY-NC-ND 4.0 terms.",
        ],
        "metrics_file": metrics_path.name,
        "figure_file": figure_path.name,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics written to {metrics_path}")
    print(f"Experiment summary written to {summary_path}")
    print(f"Metric figure written to {figure_path}")
    print(pd.DataFrame(results).to_string(index=False))
    return metrics_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    evaluate(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
```

## Appendix C. Automated Test Scripts

### C.1 Test Chapter3 Experiments

Source: `tests/test_chapter3_experiments.py`

```python
from datetime import datetime, timedelta, timezone
import unittest

import numpy as np
import pandas as pd

from src.chapter3_experiments import (
    FEATURES,
    SafeGateController,
    build_model,
    corrupt_features,
    corrupt_features_by_channel,
    one_hour_depth,
    safety_interlock,
    violates_water_limits,
)
from src.tune_gate_model import (
    THRESHOLDS,
    VALIDATION_SEED,
    make_scenario,
    tune_threshold,
)
from src.generate_philippines_rice_paddy_awd_dataset import (
    awd_policy_actions,
)
from src.predict_gate_decision import predict_sample


def sample_reading() -> pd.Series:
    return pd.Series(
        {
            "Month": "Jan",
            "Day_of_Week": "Monday",
            "Hour": 12,
            "Crop_Growth_Stage": "Vegetative/Tillering",
            "Water_Table_Depth_cm": -4.0,
            "Inundation_Duration_Days": 0.0,
            "Water_Flow_Velocity_ms": 0.1,
            "Water_pH": 6.8,
            "Soil_Moisture_VWC_Percent": 50.0,
            "Soil_Temperature_C": 25.0,
            "Vegetation_Coverage_Percent": 50.0,
            "Vegetation_Height_cm": 60.0,
        }
    )


class CorruptionTests(unittest.TestCase):
    def test_corruption_is_reproducible_and_marks_affected_rows(self) -> None:
        rows = pd.DataFrame([sample_reading()[FEATURES].to_dict()] * 40)
        for fault_type in (
            "Gaussian noise",
            "Impulse spikes",
            "Sensor drift",
            "Sensor freezing",
        ):
            first, first_mask = corrupt_features(rows, fault_type, 0.15, 123)
            second, second_mask = corrupt_features(rows, fault_type, 0.15, 123)
            pd.testing.assert_frame_equal(first, second)
            np.testing.assert_array_equal(first_mask, second_mask)
            self.assertTrue(first_mask.any())
            self.assertEqual(len(first_mask), len(rows))
            pd.testing.assert_series_equal(first["Hour"], rows["Hour"])
            pd.testing.assert_series_equal(
                first["Inundation_Duration_Days"],
                rows["Inundation_Duration_Days"],
            )

    def test_channel_fault_mask_identifies_only_selected_sensor(self) -> None:
        rows = pd.DataFrame([sample_reading()[FEATURES].to_dict()] * 40)
        corrupted, feature_mask = corrupt_features_by_channel(
            rows,
            "Sensor drift",
            0.15,
            123,
            affected_features=("Water_pH",),
        )

        self.assertTrue(feature_mask["Water_pH"].all())
        self.assertFalse(feature_mask.drop(columns="Water_pH").any().any())
        pd.testing.assert_series_equal(corrupted["Hour"], rows["Hour"])
        pd.testing.assert_series_equal(
            corrupted["Water_Table_Depth_cm"],
            rows["Water_Table_Depth_cm"],
        )


class ModelTuningTests(unittest.TestCase):
    def test_model_accepts_candidate_class_weights(self) -> None:
        model = build_model({0: 1.0, 1: 8.0})
        self.assertEqual(
            model.named_steps["classifier"].class_weight,
            {0: 1.0, 1: 8.0},
        )

    def test_seeded_scenario_is_reproducible_and_has_both_classes(self) -> None:
        first = make_scenario(VALIDATION_SEED)
        second = make_scenario(VALIDATION_SEED)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 4320)
        self.assertEqual(set(first["AWD_Gate_Action"]), {0, 1})
        flowering = first[first["Crop_Growth_Stage"] == "Reproductive/Flowering"]
        self.assertIn(1, set(flowering["AWD_Gate_Action"]))
        self.assertTrue(
            np.array_equal(
                awd_policy_actions(
                    first["Crop_Growth_Stage"].to_numpy(),
                    first["Water_Table_Depth_cm"].to_numpy(),
                    first["Soil_Moisture_VWC_Percent"].to_numpy(),
                ).astype(int),
                first["AWD_Gate_Action"].to_numpy(),
            )
        )

    def test_awd_policy_rejects_mismatched_input_shapes(self) -> None:
        with self.assertRaises(ValueError):
            awd_policy_actions(
                np.array(["Vegetative/Tillering"]),
                np.array([-15.0, -14.0]),
                np.array([25.0]),
            )

    def test_threshold_is_selected_by_validation_f2(self) -> None:
        labels = pd.Series([1, 1, 1, 0, 0, 0, 0, 0])
        probabilities = np.array([0.91, 0.61, 0.42, 0.81, 0.39, 0.21, 0.10, 0.02])

        curve, selected = tune_threshold(probabilities, labels, "test_weight")

        self.assertEqual(len(curve), len(THRESHOLDS))
        self.assertIn(selected["threshold"], THRESHOLDS)
        self.assertEqual(selected["f2_score"], float(curve["f2_score"].max()))
        self.assertEqual(selected["class_weight"], "test_weight")

    def test_predict_sample_uses_explicit_threshold_when_provided(self) -> None:
        class StubModel:
            @staticmethod
            def predict_proba(sample: pd.DataFrame) -> np.ndarray:
                return np.tile(np.array([[0.48, 0.52]]), (len(sample), 1))

            @staticmethod
            def predict(sample: pd.DataFrame) -> np.ndarray:
                return np.ones(len(sample), dtype=int)

        sample = pd.DataFrame([{"feature": 1}])
        model = StubModel()

        self.assertEqual(predict_sample(model, sample, 0.53), (0, 0.52))
        self.assertEqual(predict_sample(model, sample, 0.51), (1, 0.52))
        self.assertEqual(predict_sample(model, sample), (1, 0.52))
        with self.assertRaises(ValueError):
            predict_sample(model, sample, 1.1)


class SafetyControllerTests(unittest.TestCase):
    def test_model_request_is_advisory_and_policy_controls_valid_readings(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        reading = sample_reading()
        reading["Water_Table_Depth_cm"] = -15.0
        result = SafeGateController().step(
            reading,
            now,
            now,
            requested_open=False,
        )

        self.assertTrue(result.gate_open)
        self.assertFalse(result.safe_mode)
        self.assertFalse(result.model_agrees_with_policy)

    def test_noncritical_fault_uses_policy_in_degraded_mode(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = SafeGateController().step(
            sample_reading(),
            now,
            now,
            requested_open=True,
            faulty_features=("Water_pH",),
        )

        self.assertFalse(result.safe_mode)
        self.assertTrue(result.degraded_mode)
        self.assertEqual(result.faulted_features, ("Water_pH",))
        self.assertTrue(result.reason.startswith("advisory_sensor_fault:"))

    def test_critical_water_depth_fault_enters_safe_mode(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        controller = SafeGateController(initial_gate_open=False)
        reading = sample_reading()
        reading["Water_Table_Depth_cm"] = -15.0
        result = controller.step(
            reading,
            now,
            now,
            requested_open=True,
            faulty_features=("Water_Table_Depth_cm",),
        )

        self.assertTrue(result.safe_mode)
        self.assertEqual(
            result.reason,
            "critical_sensor_fault:Water_Table_Depth_cm",
        )
        self.assertFalse(result.gate_open)

    def test_moisture_fault_is_noncritical_during_flowering(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        reading = sample_reading()
        reading["Crop_Growth_Stage"] = "Reproductive/Flowering"
        reading["Water_Table_Depth_cm"] = 0.2
        result = SafeGateController().step(
            reading,
            now,
            now,
            faulty_features=("Soil_Moisture_VWC_Percent",),
        )

        self.assertFalse(result.safe_mode)
        self.assertTrue(result.degraded_mode)
        self.assertTrue(result.gate_open)

    def test_stale_reading_holds_last_gate_command(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        controller = SafeGateController(initial_gate_open=True)
        result = controller.step(
            sample_reading(),
            now - timedelta(seconds=3601),
            now,
            requested_open=False,
        )
        self.assertTrue(result.safe_mode)
        self.assertEqual(result.reason, "stale_reading")
        self.assertTrue(result.gate_open)

    def test_detected_corruption_holds_last_gate_command(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        controller = SafeGateController(initial_gate_open=False)
        result = controller.step(
            sample_reading(),
            now,
            now,
            fault_detected=True,
            requested_open=True,
        )
        self.assertTrue(result.safe_mode)
        self.assertEqual(result.reason, "sensor_fault_flag")
        self.assertFalse(result.gate_open)

    def test_repeated_identical_numeric_readings_trigger_safe_mode(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        controller = SafeGateController(initial_gate_open=True)
        reading = sample_reading()
        reading["Water_Table_Depth_cm"] = -15.0
        first = controller.step(reading, now, now, requested_open=True)
        second = controller.step(reading, now, now, requested_open=True)
        third = controller.step(reading, now, now, requested_open=False)

        self.assertFalse(first.safe_mode)
        self.assertFalse(second.safe_mode)
        self.assertTrue(third.safe_mode)
        self.assertTrue(third.reason.startswith("critical_sensor_fault:"))
        self.assertTrue(third.gate_open)

    def test_single_frozen_noncritical_sensor_uses_degraded_mode(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        controller = SafeGateController()
        results = []
        for step in range(3):
            reading = sample_reading()
            reading["Water_Table_Depth_cm"] = -4.0 + step
            reading["Water_Flow_Velocity_ms"] = 0.1 + step * 0.01
            reading["Soil_Moisture_VWC_Percent"] = 50.0 + step
            reading["Soil_Temperature_C"] = 25.0 + step * 0.1
            reading["Vegetation_Coverage_Percent"] = 50.0 + step
            reading["Vegetation_Height_cm"] = 60.0 + step
            results.append(
                controller.step(
                    reading,
                    now,
                    now,
                    requested_open=False,
                )
            )
        self.assertFalse(results[0].safe_mode)
        self.assertFalse(results[1].safe_mode)
        self.assertFalse(results[2].safe_mode)
        self.assertTrue(results[2].degraded_mode)
        self.assertIn("Water_pH", results[2].faulted_features)

    def test_missing_or_out_of_range_sensor_enters_safe_mode(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        controller = SafeGateController(initial_gate_open=False)
        missing = sample_reading().drop("Water_pH")
        missing_result = controller.step(missing, now, now, requested_open=True)
        self.assertTrue(missing_result.safe_mode)
        self.assertEqual(missing_result.reason, "missing:Water_pH")

        invalid = sample_reading()
        invalid["Water_pH"] = 9.0
        invalid_result = controller.step(invalid, now, now, requested_open=True)
        self.assertTrue(invalid_result.safe_mode)
        self.assertEqual(invalid_result.reason, "out_of_range:Water_pH")

        non_finite = sample_reading()
        non_finite["Water_pH"] = np.nan
        non_finite_result = controller.step(
            non_finite, now, now, requested_open=True
        )
        self.assertTrue(non_finite_result.safe_mode)
        self.assertEqual(non_finite_result.reason, "non_finite:Water_pH")

    def test_invalid_timestamp_and_non_integer_hour_enter_safe_mode(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        controller = SafeGateController()
        naive = datetime(2024, 1, 1)
        timestamp_result = controller.step(
            sample_reading(), naive, now, requested_open=True
        )
        self.assertTrue(timestamp_result.safe_mode)
        self.assertEqual(timestamp_result.reason, "invalid_timestamp")

        reading = sample_reading()
        reading["Hour"] = 12.5
        hour_result = controller.step(reading, now, now, requested_open=True)
        self.assertTrue(hour_result.safe_mode)
        self.assertEqual(hour_result.reason, "invalid:Hour")

    def test_interlock_prevents_one_step_lower_and_upper_boundary_crossings(self) -> None:
        flowering = sample_reading()
        flowering["Crop_Growth_Stage"] = "Reproductive/Flowering"
        flowering["Water_Table_Depth_cm"] = 0.2
        self.assertTrue(safety_interlock(False, flowering))

        near_upper_limit = sample_reading()
        near_upper_limit["Water_Table_Depth_cm"] = 4.5
        self.assertFalse(safety_interlock(True, near_upper_limit))

        self.assertFalse(violates_water_limits(0.1, "Reproductive/Flowering"))
        self.assertTrue(violates_water_limits(-0.1, "Reproductive/Flowering"))
        self.assertTrue(violates_water_limits(-15.1, "Vegetative/Tillering"))
        self.assertTrue(violates_water_limits(5.1, "Vegetative/Tillering"))
        self.assertEqual(one_hour_depth(2.0, True), 3.0)

    def test_interlock_selects_lower_violation_when_no_action_is_fully_safe(self) -> None:
        reading = sample_reading()
        reading["Crop_Growth_Stage"] = "Reproductive/Flowering"
        reading["Water_Table_Depth_cm"] = -2.0

        self.assertTrue(safety_interlock(False, reading))

    def test_interlock_honors_available_safe_action_across_depth_range(self) -> None:
        for stage in (
            "Transplanting",
            "Vegetative/Tillering",
            "Reproductive/Flowering",
            "Ripening/Maturation",
        ):
            for depth in np.linspace(-20.0, 10.0, 301):
                reading = sample_reading()
                reading["Crop_Growth_Stage"] = stage
                reading["Water_Table_Depth_cm"] = depth
                selected = safety_interlock(False, reading)
                selected_projection = one_hour_depth(depth, selected)
                other_projection = one_hour_depth(depth, not selected)
                if not violates_water_limits(other_projection, stage):
                    self.assertFalse(
                        violates_water_limits(selected_projection, stage)
                    )


if __name__ == "__main__":
    unittest.main()
```

### C.2 Test External Irrigation Data

Source: `tests/test_external_irrigation_data.py`

```python
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.evaluate_external_irrigation_data import (
    Stream,
    evaluate,
    latest_value,
    load_stream,
    summarize_relay_observations,
    summarize_stream,
)


class ExternalIrrigationDataTests(unittest.TestCase):
    def test_load_stream_parses_records_and_counts_duplicate_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "readings.txt"
            path.write_text(
                "moisture,time_stamp\n"
                "44.5,12-Jul-2022 00:00:00\n"
                "45.0,12-Jul-2022 00:00:00\n",
                encoding="utf-8",
            )

            stream = load_stream(path, "moisture")

        self.assertEqual(stream.values, [44.5, 45.0])
        self.assertEqual(stream.duplicate_timestamps, 1)

    def test_load_stream_rejects_out_of_order_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "readings.txt"
            path.write_text(
                "relay,time_stamp\n"
                "0,12-Jul-2022 00:05:00\n"
                "1,12-Jul-2022 00:00:00\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "ascending order"):
                load_stream(path, "relay")

    def test_latest_value_enforces_maximum_age_and_never_uses_future_data(self) -> None:
        stream = Stream(
            name="moisture",
            value_column="moisture",
            timestamps=[
                datetime(2022, 7, 12, 0, 0),
                datetime(2022, 7, 12, 0, 20),
            ],
            values=[50.0, 55.0],
            duplicate_timestamps=0,
        )

        self.assertIsNone(
            latest_value(stream, datetime(2022, 7, 11, 23, 59), 1800)
        )
        self.assertEqual(
            latest_value(stream, datetime(2022, 7, 12, 0, 25), 600), 55.0
        )
        self.assertIsNone(
            latest_value(stream, datetime(2022, 7, 12, 0, 51), 1800)
        )

    def test_relay_summary_pairs_only_recent_preceding_moisture(self) -> None:
        moisture = Stream(
            name="moisture",
            value_column="moisture",
            timestamps=[
                datetime(2022, 7, 12, 0, 0),
                datetime(2022, 7, 12, 1, 0),
            ],
            values=[40.0, 60.0],
            duplicate_timestamps=0,
        )
        relay = Stream(
            name="relay",
            value_column="relay",
            timestamps=[
                datetime(2022, 7, 12, 0, 10),
                datetime(2022, 7, 12, 0, 20),
                datetime(2022, 7, 12, 1, 40),
            ],
            values=[0.0, 1.0, 1.0],
            duplicate_timestamps=0,
        )

        rows = summarize_relay_observations(relay, moisture, 30 * 60)

        self.assertEqual(rows[0]["paired_moisture_records"], 1)
        self.assertEqual(rows[0]["moisture_median"], 40.0)
        self.assertEqual(rows[1]["paired_moisture_records"], 1)
        self.assertEqual(rows[1]["moisture_median"], 40.0)

    def test_stream_summary_reports_gaps_and_range(self) -> None:
        stream = Stream(
            name="relay",
            value_column="relay",
            timestamps=[
                datetime(2022, 7, 12, 0, 0),
                datetime(2022, 7, 12, 0, 5),
                datetime(2022, 7, 12, 1, 0),
            ],
            values=[0.0, 1.0, 0.0],
            duplicate_timestamps=0,
        )

        summary = summarize_stream(stream, expected_interval_seconds=300)

        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["minimum"], 0.0)
        self.assertEqual(summary["maximum"], 1.0)
        self.assertEqual(summary["gaps_over_1_5x_expected"], 1)
        self.assertEqual(summary["maximum_gap_seconds"], 3300.0)

    def test_evaluate_writes_aggregate_report_without_touching_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            output_dir = root / "reports"
            data_dir.mkdir()
            (data_dir / "D_moisture.txt").write_text(
                "moisture,time_stamp\n"
                "40,12-Jul-2022 00:00:00\n"
                "45,12-Jul-2022 00:20:00\n",
                encoding="utf-8",
            )
            (data_dir / "D_flowmeter.txt").write_text(
                "litres,time_stamp\n"
                "0,12-Jul-2022 00:00:00\n"
                "1,12-Jul-2022 00:05:00\n",
                encoding="utf-8",
            )
            (data_dir / "D_valve.txt").write_text(
                "relay,time_stamp\n"
                "0,12-Jul-2022 00:00:00\n"
                "1,12-Jul-2022 00:05:00\n",
                encoding="utf-8",
            )

            report_path, streams_path = evaluate(data_dir, output_dir)

            self.assertTrue(report_path.is_file())
            self.assertTrue(streams_path.is_file())
            self.assertIn(
                "No model was retrained or changed",
                report_path.read_text(encoding="utf-8"),
            )
            self.assertTrue((data_dir / "D_valve.txt").is_file())


if __name__ == "__main__":
    unittest.main()
```

### C.3 Test External Soil Moisture Forecasting

Source: `tests/test_external_soil_moisture_forecasting.py`

```python
import unittest

import pandas as pd

from experiments.irrigation_forecasting.evaluate import (
    FEATURE_COLUMNS,
    build_forecasting_rows,
    split_by_time,
)


class SoilMoistureForecastingTests(unittest.TestCase):
    def test_feature_rows_use_exact_lag_dates_and_skip_gaps(self) -> None:
        dates = pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
                "2024-01-07",
                "2024-01-08",
                "2024-01-10",
            ]
        )
        readings = pd.DataFrame(
            {
                "consortium": ["consortium0"] * len(dates),
                "datastream_name": ["stream-a"] * len(dates),
                "result_time": dates,
                "result": range(len(dates)),
                "sensor_type": ["Water Content"] * len(dates),
            }
        )

        rows = build_forecasting_rows(readings)

        self.assertEqual(
            list(rows.columns),
            ["consortium", "datastream_name", "target_time", "target", *FEATURE_COLUMNS],
        )
        self.assertEqual(
            list(rows["target_time"]),
            list(pd.to_datetime(["2024-01-08"])),
        )
        self.assertEqual(rows.iloc[0]["lag_1_day"], 6)
        self.assertEqual(rows.iloc[0]["lag_2_days"], 5)
        self.assertEqual(rows.iloc[0]["lag_7_days"], 0)

    def test_splits_are_disjoint_and_chronological(self) -> None:
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        rows = pd.DataFrame(
            {
                "target_time": dates,
                "datastream_name": ["stream-a"] * len(dates),
                "target": range(len(dates)),
            }
        )

        train, validation, test = split_by_time(rows)

        self.assertLess(train["target_time"].max(), validation["target_time"].min())
        self.assertLess(
            validation["target_time"].max(), test["target_time"].min()
        )
        self.assertEqual(len(train) + len(validation) + len(test), len(rows))


if __name__ == "__main__":
    unittest.main()
```

## Appendix D. Jupyter Notebook Code Cells

The following listings reproduce every code cell from the project notebooks in notebook order. Cell outputs are intentionally omitted.

### D.1 Chapter2 Methodology

### D.1.1 Cell 2 (`feature-parameter-summary`)

Source: `chapter2_methodology.ipynb`

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/philippines_rice_paddy_awd_dataset.csv")
REPORT_DIR = Path("reports")
MARKDOWN_PATH = REPORT_DIR / "chapter2_feature_parameter_summary.md"
LATEX_PATH = REPORT_DIR / "chapter2_feature_parameter_summary.tex"
FIGURE_PATH = REPORT_DIR / "chapter2_feature_ranges.png"

dataset = pd.read_csv(DATA_PATH)
feature_metadata = {
    "Month": ("Month", "calendar month", "Context", "Seasonal context; wet-season months use the higher synthetic rainfall factor."),
    "Day_of_Week": ("Day of week", "day name", "Context", "Temporal context; no direct ecological threshold is assigned."),
    "Hour": ("Hour", "h", "Context", "Diurnal context for the synthetic sensor stream."),
    "Crop_Growth_Stage": ("Crop growth stage", "stage", "Agronomic context", "Flowering safety takes precedence: water remains above 0 cm and the gate opens below 2 cm."),
    "Water_Table_Depth_cm": ("Water-table depth", "cm", "Hydrology", "AWD variable; outside flowering the gate opens at <= -15 cm, while flowering maintains flooding above 0 cm."),
    "Inundation_Duration_Days": ("Inundation duration", "days", "Hydrology", "Simulated O. hupensis habitat duration; >14 days contributes to High risk and >3 days to Moderate risk."),
    "Water_Flow_Velocity_ms": ("Water-flow velocity", "m/s", "Hydrology", "Hydraulic context; no independent simulated snail-risk threshold is configured."),
    "Water_pH": ("Water pH", "pH", "Water quality", "Simulated habitat risk uses 5.5-7.9 for High risk and 5.3-8.1 for Moderate risk."),
    "Soil_Moisture_VWC_Percent": ("Soil moisture", "% VWC", "Soil", "Outside flowering, AWD opens at <=25%; habitat-risk conditions require 20-80%."),
    "Soil_Temperature_C": ("Soil temperature", "deg C", "Soil/ecology", "Simulated habitat risk uses 15-30 deg C for High risk and 15-32 deg C for Moderate risk."),
    "Vegetation_Coverage_Percent": ("Vegetation coverage", "%", "Vegetation", "Crop-canopy context; no independent threshold is configured."),
    "Vegetation_Height_cm": ("Vegetation height", "cm", "Vegetation", "Crop-growth context; no independent threshold is configured."),
}

missing = sorted(set(feature_metadata) - set(dataset.columns))
if missing:
    raise ValueError(f"Missing expected feature columns: {missing}")

numeric_features = dataset[list(feature_metadata)].select_dtypes(include="number").columns
statistics = dataset[numeric_features].agg(["min", "max", "mean", "std"]).T
statistics = statistics.rename(columns={"min": "Minimum", "max": "Maximum", "mean": "Mean", "std": "SD"})
rows = []
for column, (name, unit, domain, logic) in feature_metadata.items():
    row = {"Feature": name, "Unit": unit, "Domain": domain, "Minimum": "-", "Maximum": "-", "Mean": "-", "SD": "-", "AWD / O. hupensis safety logic": logic}
    if column in statistics.index:
        row.update({key: f"{value:.4f}" for key, value in statistics.loc[column].items()})
    rows.append(row)
summary = pd.DataFrame(rows)

markdown = ("# Chapter 2 Feature Parameter Summary\n\n" "*Synthetic dataset assumptions; `Snail_Habitat_Risk` and `AWD_Gate_Action` are excluded labels.*\n\n" + summary.to_markdown(index=False) + "\n")
latex = summary.to_latex(index=False, escape=True, longtable=True, caption="Feature parameter summary and configured ecological safety logic.", label="tab:chapter2-feature-parameters", column_format="p{0.17\\textwidth}p{0.08\\textwidth}p{0.10\\textwidth}rrrrp{0.42\\textwidth}")

print(markdown)
print("\nLaTeX representation:\n")
print(latex)

plot_data = statistics.loc[list(numeric_features)].copy()
plot_data["Feature"] = [feature_metadata[feature][0] for feature in plot_data.index]
plot_data = plot_data.reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
ax.errorbar(plot_data["Mean"], plot_data["Feature"], xerr=[plot_data["Mean"] - plot_data["Minimum"], plot_data["Maximum"] - plot_data["Mean"]], fmt="o", color="#176b87", ecolor="#7aa9b8", capsize=4)
ax.set_title("Numeric Feature Ranges in the Synthetic AWD Dataset")
ax.set_xlabel("Value in feature-specific units")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_PATH.write_text(markdown, encoding="utf-8")
LATEX_PATH.write_text(latex, encoding="utf-8")
fig.savefig(FIGURE_PATH, bbox_inches="tight")
plt.show()
print(f"\nWrote {MARKDOWN_PATH}, {LATEX_PATH}, and {FIGURE_PATH}")
```

### D.2 Chapter3

### D.2.1 Cell 1 (`9f8aca35`)

Source: `chapter3.ipynb`

```python
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load pre-generated robustness results
df_results = pd.read_csv("data/sensor_robustness_results.csv")
print("Data loaded successfully!")
display(df_results.head())

# Load baseline model metrics
with open("models/model_metrics_random_forest.json", "r") as f:
    baseline_metrics = json.load(f)

print("Baseline Metrics:", baseline_metrics)
```

### D.2.2 Cell 2 (`7c60aeeb`)

Source: `chapter3.ipynb`

```python
import joblib

# Load trained model directly
rf_model = joblib.load("models/gate_model_random_forest.joblib")
print("Model loaded into notebook memory:", type(rf_model))
```

### D.2.3 Cell 3 (`b40ca3c9`)

Source: `chapter3.ipynb`

```python
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 5), dpi=300)

if (
    "noise_level" in df_results.columns
    and "f1_score" in df_results.columns
    and "fault_type" in df_results.columns
):
    sns.lineplot(
        data=df_results,
        x="noise_level",
        y="f1_score",
        hue="fault_type",
        marker="o",
        linewidth=2.5,
    )

    plt.title(
        "Random Forest Robustness Under Sensor Corruptions",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Noise Intensity Level (%)")
    plt.ylabel("F1-Score")
    plt.legend(title="Fault Type")
    plt.tight_layout()

    plt.savefig("reports/Figure_4_2_Degradation_Curves.png", dpi=300)
    plt.show()
else:
    print("Columns available in CSV:", df_results.columns.tolist())
```

## Appendix E. ESP32/Wokwi Firmware

### E.1 ESP32 Controller Demonstrator

Source: `wokwi/src/main.cpp`

```cpp
#include <Arduino.h>
#include <ESP32Servo.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace {

constexpr uint8_t kGateOpenPin = 18;
constexpr uint8_t kGateClosedPin = 19;
constexpr uint8_t kSafeModePin = 21;
constexpr uint8_t kGateServoPin = 23;
constexpr int kGateClosedAngle = 0;
constexpr int kGateOpenAngle = 90;
constexpr float kMinimumDepthCm = -20.0F;
constexpr float kMaximumDepthCm = 10.0F;
constexpr float kAwdMinimumDepthCm = -15.0F;
constexpr float kFloweringMinimumDepthCm = 0.0F;
constexpr float kMaximumDepthLimitCm = 5.0F;
constexpr float kOpenDepthChangeCmPerHour = 1.0F;
constexpr float kClosedDepthChangeCmPerHour = -0.5F;
constexpr long kMaximumReadingAgeMs = 3600000L;
constexpr size_t kLineCapacity = 128;
constexpr uint8_t kMaximumTokens = 7;

Servo gateServo;

struct Reading {
  char stage[16];
  float depthCm;
  bool requestedOpen;
  long ageMs;
  bool faultFlag;
};

bool gateOpen = false;
bool safeMode = false;
bool havePreviousDepth = false;
float previousDepthCm = 0.0F;
uint8_t identicalDepthCount = 0;
char inputLine[kLineCapacity];
size_t inputLength = 0;
bool inputOverflow = false;

void updateIndicators() {
  digitalWrite(kGateOpenPin, gateOpen ? HIGH : LOW);
  digitalWrite(kGateClosedPin, gateOpen ? LOW : HIGH);
  digitalWrite(kSafeModePin, safeMode ? HIGH : LOW);
  gateServo.write(gateOpen ? kGateOpenAngle : kGateClosedAngle);
}

void enterSafeMode(const char *reason) {
  safeMode = true;
  updateIndicators();
  Serial.print("RESULT gate=");
  Serial.print(gateOpen ? "OPEN" : "CLOSED");
  Serial.print(" safe=1 reason=");
  Serial.println(reason);
}

bool parseFloatToken(const char *token, float &value) {
  if (token == nullptr || *token == '\0') {
    return false;
  }
  char *end = nullptr;
  value = strtof(token, &end);
  return end != token && *end == '\0';
}

bool parseLongToken(const char *token, long &value) {
  if (token == nullptr || *token == '\0') {
    return false;
  }
  char *end = nullptr;
  value = strtol(token, &end, 10);
  return end != token && *end == '\0';
}

bool parseBinaryToken(const char *token, bool &value) {
  long parsed = 0;
  if (!parseLongToken(token, parsed) || (parsed != 0 && parsed != 1)) {
    return false;
  }
  value = parsed == 1;
  return true;
}

bool parseReading(char **tokens, uint8_t count, Reading &reading) {
  if (count != 6) {
    return false;
  }
  if (strlen(tokens[1]) >= sizeof(reading.stage)) {
    return false;
  }
  strcpy(reading.stage, tokens[1]);
  return parseFloatToken(tokens[2], reading.depthCm) &&
         parseBinaryToken(tokens[3], reading.requestedOpen) &&
         parseLongToken(tokens[4], reading.ageMs) &&
         parseBinaryToken(tokens[5], reading.faultFlag);
}

bool isKnownStage(const char *stage) {
  return strcmp(stage, "TRANSPLANT") == 0 ||
         strcmp(stage, "VEG") == 0 ||
         strcmp(stage, "FLOWER") == 0 ||
         strcmp(stage, "RIPEN") == 0;
}

bool isFlowering(const char *stage) {
  return strcmp(stage, "FLOWER") == 0;
}

bool passesValidityChecks(const Reading &reading, const char *&reason) {
  if (reading.faultFlag) {
    reason = "sensor_fault_flag";
    return false;
  }
  if (reading.ageMs < 0 || reading.ageMs > kMaximumReadingAgeMs) {
    reason = "stale_reading";
    return false;
  }
  if (!isfinite(reading.depthCm)) {
    reason = "non_finite:depth_cm";
    return false;
  }
  if (reading.depthCm < kMinimumDepthCm ||
      reading.depthCm > kMaximumDepthCm) {
    reason = "out_of_range:depth_cm";
    return false;
  }
  if (!isKnownStage(reading.stage)) {
    reason = "invalid:crop_stage";
    return false;
  }
  return true;
}

bool isFrozenDepth(float depthCm) {
  if (!havePreviousDepth) {
    havePreviousDepth = true;
    previousDepthCm = depthCm;
    identicalDepthCount = 0;
    return false;
  }
  if (depthCm == previousDepthCm) {
    if (identicalDepthCount < UINT8_MAX) {
      ++identicalDepthCount;
    }
  } else {
    identicalDepthCount = 0;
  }
  previousDepthCm = depthCm;
  return identicalDepthCount >= 2;
}

bool applyInterlock(bool requestedOpen, const Reading &reading) {
  const float minimumDepth =
      isFlowering(reading.stage) ? kFloweringMinimumDepthCm : kAwdMinimumDepthCm;
  const float openProjection =
      reading.depthCm + kOpenDepthChangeCmPerHour;
  const float closeProjection =
      reading.depthCm + kClosedDepthChangeCmPerHour;

  if (openProjection > kMaximumDepthLimitCm) {
    return false;
  }
  if (closeProjection < minimumDepth) {
    return true;
  }
  return requestedOpen;
}

void processSample(char **tokens, uint8_t count) {
  Reading reading{};
  if (!parseReading(tokens, count, reading)) {
    enterSafeMode("invalid_sample_format");
    return;
  }

  const char *reason = nullptr;
  if (!passesValidityChecks(reading, reason)) {
    enterSafeMode(reason);
    return;
  }
  if (isFrozenDepth(reading.depthCm)) {
    enterSafeMode("frozen_depth");
    return;
  }

  gateOpen = applyInterlock(reading.requestedOpen, reading);
  safeMode = false;
  updateIndicators();
  Serial.print("RESULT gate=");
  Serial.print(gateOpen ? "OPEN" : "CLOSED");
  Serial.println(" safe=0 reason=ok");
}

void resetController() {
  gateOpen = false;
  safeMode = false;
  havePreviousDepth = false;
  previousDepthCm = 0.0F;
  identicalDepthCount = 0;
  updateIndicators();
  Serial.println("RESET gate=CLOSED safe=0");
}

void printHelp() {
  Serial.println("Commands:");
  Serial.println("  SAMPLE <stage> <depth_cm> <request_open:0|1> <age_ms> <fault:0|1>");
  Serial.println("  stage: TRANSPLANT | VEG | FLOWER | RIPEN");
  Serial.println("  RESET");
  Serial.println("Example: SAMPLE VEG -15 1 0 0");
  Serial.println("The requested action is a test input; the Random Forest is not running on the ESP32.");
}

void processLine(char *line) {
  char *tokens[kMaximumTokens] = {};
  uint8_t count = 0;
  char *savePointer = nullptr;
  for (char *token = strtok_r(line, " \t", &savePointer);
       token != nullptr;
       token = strtok_r(nullptr, " \t", &savePointer)) {
    if (count >= kMaximumTokens) {
      enterSafeMode("too_many_fields");
      return;
    }
    tokens[count++] = token;
  }
  if (count == 0) {
    return;
  }
  if (strcmp(tokens[0], "HELP") == 0 && count == 1) {
    printHelp();
  } else if (strcmp(tokens[0], "RESET") == 0 && count == 1) {
    resetController();
  } else if (strcmp(tokens[0], "SAMPLE") == 0) {
    processSample(tokens, count);
  } else {
    enterSafeMode("unknown_command");
  }
}

void runDemoCommand(
    const char *label,
    const char *command,
    bool expectedGateOpen,
    bool expectedSafeMode) {
  char line[kLineCapacity];
  strncpy(line, command, sizeof(line) - 1);
  line[sizeof(line) - 1] = '\0';

  Serial.print("DEMO_CASE ");
  Serial.print(label);
  Serial.print(" command=\"");
  Serial.print(command);
  Serial.println("\"");
  processLine(line);

  const bool passed =
      gateOpen == expectedGateOpen && safeMode == expectedSafeMode;
  Serial.print("DEMO_CHECK ");
  Serial.print(passed ? "PASS" : "FAIL");
  Serial.print(" expected_gate=");
  Serial.print(expectedGateOpen ? "OPEN" : "CLOSED");
  Serial.print(" expected_safe=");
  Serial.print(expectedSafeMode ? "1" : "0");
  Serial.print(" actual_gate=");
  Serial.print(gateOpen ? "OPEN" : "CLOSED");
  Serial.print(" actual_safe=");
  Serial.println(safeMode ? "1" : "0");
  delay(800);
}

void runStartupDemo() {
  Serial.println("DEMO_BEGIN startup_controller_checks");
  runDemoCommand("reset", "RESET", false, false);
  runDemoCommand("low_depth_requests_open", "SAMPLE VEG -15 1 0 0", true, false);
  runDemoCommand("upper_limit_blocks_open", "SAMPLE VEG 4.5 1 0 0", false, false);
  runDemoCommand("flowering_minimum", "SAMPLE FLOWER 0.2 0 0 0", true, false);
  runDemoCommand("reset_before_stale_reading", "RESET", false, false);
  runDemoCommand("stale_reading_safe_mode", "SAMPLE VEG -4 1 3600001 0", false, true);
  runDemoCommand("reset_before_fault_flag", "RESET", false, false);
  runDemoCommand("fault_flag_safe_mode", "SAMPLE VEG -4 1 0 1", false, true);
  runDemoCommand("reset_before_freeze_check", "RESET", false, false);
  runDemoCommand("freeze_sample_1", "SAMPLE VEG -4 0 0 0", false, false);
  runDemoCommand("freeze_sample_2", "SAMPLE VEG -4 0 0 0", false, false);
  runDemoCommand("freeze_sample_3_safe_mode", "SAMPLE VEG -4 0 0 0", false, true);
  Serial.println("DEMO_END startup_controller_checks");
}

void readSerialLines() {
  while (Serial.available() > 0) {
    const char incoming = static_cast<char>(Serial.read());
    if (incoming == '\r') {
      continue;
    }
    if (incoming == '\n') {
      if (inputOverflow) {
        enterSafeMode("command_too_long");
      } else {
        inputLine[inputLength] = '\0';
        processLine(inputLine);
      }
      inputLength = 0;
      inputOverflow = false;
      continue;
    }
    if (inputLength + 1 < sizeof(inputLine)) {
      inputLine[inputLength++] = incoming;
    } else {
      inputOverflow = true;
    }
  }
}

}  // namespace

void setup() {
  pinMode(kGateOpenPin, OUTPUT);
  pinMode(kGateClosedPin, OUTPUT);
  pinMode(kSafeModePin, OUTPUT);
  Serial.begin(115200);
  gateServo.attach(kGateServoPin);
  resetController();
  printHelp();
  runStartupDemo();
}

void loop() {
  readSerialLines();
}
```
