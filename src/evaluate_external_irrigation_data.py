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
