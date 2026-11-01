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
