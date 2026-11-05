"""Generate supplementary Chapter 2 and Chapter 3 figures from saved summaries."""

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
DEFAULT_CHAPTER2_OUTPUT_DIR = ROOT / "reports" / "chapter2_methodology"
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
    metrics = "#f1f5f9"
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
    add_box(ax, (0.65, 0.24), "Model-policy agreement\nand prediction metrics", metrics, 0.2, 0.15)
    add_arrow(ax, (0.66, 0.69), (0.47, 0.32), color="#a16207", linestyle="--")
    add_arrow(ax, (0.515, 0.24), (0.55, 0.24), color="#a16207", linestyle="--")
    add_arrow(ax, (0.69, 0.37), (0.67, 0.32), color="#a16207", linestyle="--")

    add_box(ax, (0.14, 0.24), "Critical / invalid input:\nsafe mode; hold last command", safe, 0.23, 0.15)
    add_arrow(ax, (0.12, 0.36), (0.14, 0.33), color="#b91c1c")

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
    parser.add_argument(
        "--chapter2-output-dir", type=Path, default=DEFAULT_CHAPTER2_OUTPUT_DIR
    )
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.chapter2_output_dir.mkdir(parents=True, exist_ok=True)
    plot_evaluation_architecture(
        args.output_dir / "Figure_3_7_evaluation_architecture.png"
    )
    plot_external_audit(
        args.audit_dir,
        args.chapter2_output_dir / "Figure_2_1_external_data_audit.png",
    )
    print(f"Supplementary Chapter 3 figure written to {args.output_dir}")
    print(f"Chapter 2 audit figure written to {args.chapter2_output_dir}")


if __name__ == "__main__":
    main()
