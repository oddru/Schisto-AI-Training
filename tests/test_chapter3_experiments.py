from datetime import datetime, timedelta, timezone
import unittest

import numpy as np
import pandas as pd

from src.chapter3_experiments import (
    FEATURES,
    SafeGateController,
    corrupt_features,
    one_hour_depth,
    safety_interlock,
    violates_water_limits,
)


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


class SafetyControllerTests(unittest.TestCase):
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
        first = controller.step(reading, now, now, requested_open=True)
        second = controller.step(reading, now, now, requested_open=True)
        third = controller.step(reading, now, now, requested_open=False)

        self.assertFalse(first.safe_mode)
        self.assertFalse(second.safe_mode)
        self.assertTrue(third.safe_mode)
        self.assertEqual(third.reason, "frozen_readings")
        self.assertTrue(third.gate_open)

    def test_single_frozen_sensor_triggers_safe_mode(self) -> None:
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
        self.assertTrue(results[2].safe_mode)
        self.assertEqual(results[2].reason, "frozen_readings")

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


if __name__ == "__main__":
    unittest.main()
