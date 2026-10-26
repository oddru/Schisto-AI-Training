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
