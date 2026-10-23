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
    corrupt_features,
    evaluate_actuation,
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
                corrupted, fault_mask = corrupt_features(
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
        "random_state": RANDOM_STATE,
        "same_generator_assumptions_across_seeds": True,
        "robustness_repeats": repeats,
        "fallback_fault_mask": (
            "Oracle injection mask; used to evaluate fallback behavior, "
            "not a validated fault detector."
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
        "assumptions": output_dir / "tuning_assumptions.json",
        "model": output_dir / "gate_model_tuned.joblib",
        "figures": output_dir / "Figure_3_2_validation_thresholds.png",
        "test_figure": output_dir / "Figure_3_3_locked_test_comparison.png",
        "robustness_figure": output_dir / "Figure_3_4_robustness_curves.png",
        "safety_figure": output_dir / "Figure_3_5_awd_safety.png",
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
    axes[1].set_title("Interlock and fallback")
    axes[1].legend(fontsize=7)
    figure.savefig(output_dir / "Figure_3_5_awd_safety.png", dpi=200)
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
