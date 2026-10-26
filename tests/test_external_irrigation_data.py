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
