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
