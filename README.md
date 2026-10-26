# Schisto-AI-Training

This repository contains a synthetic Philippine lowland rice-paddy IoT benchmark for AWD gate control and simulated snail-habitat risk analysis.

## Active dataset

The active dataset is `data/philippines_rice_paddy_awd_dataset.csv`. It contains hourly readings with:

- temporal features: `Month`, `Day_of_Week`, `Hour`
- crop stage: `Crop_Growth_Stage`
- hydrology: `Water_Table_Depth_cm`, `Inundation_Duration_Days`, `Water_Flow_Velocity_ms`, `Water_pH`
- soil: `Soil_Moisture_VWC_Percent`, `Soil_Temperature_C`
- vegetation: `Vegetation_Coverage_Percent`, `Vegetation_Height_cm`
- labels: `Snail_Habitat_Risk`, `AWD_Gate_Action`

`AWD_Gate_Action` is the Random Forest target:

- `1` = gate open / reflood
- `0` = gate closed / maintain state

`Snail_Habitat_Risk` is a separate rule-derived categorical label and is not used as a model feature, avoiding target leakage.

The data and labels are synthetic. They encode the supplied AWD and habitat-risk assumptions; they are not field observations or proof of disease-transmission prevention.

The single-sample Random Forest predictor reports an advisory classification, not an irrigation command. In the Python controller experiment, a separate explicit AWD rule controls valid readings and the Random Forest is used only for comparison.

## Dataset generation

`src/generate_philippines_rice_paddy_awd_dataset.py` creates a reproducible hourly stream, couples soil moisture to water depth, applies the flowering flooding override, calculates the risk labels, and validates the configured physical and logical ranges.

```bash
python src/generate_philippines_rice_paddy_awd_dataset.py --output data/philippines_rice_paddy_awd_dataset.csv
```

## Random Forest training

`src/train_gate_model.py` uses one-hot encoding for categorical features and a 400-tree balanced Random Forest for `AWD_Gate_Action`.

```bash
python src/train_gate_model.py --dataset data/philippines_rice_paddy_awd_dataset.csv --output-dir models
```

## Predict one new sample

```bash
python src/predict_gate_decision.py --model-path models/gate_model_random_forest.joblib \
  --month Jan --day-of-week Monday --hour 12 \
  --crop-growth-stage Vegetative/Tillering \
  --water-table-depth-cm -15 --inundation-duration-days 0 \
  --water-flow-velocity-ms 0.10 --water-ph 6.8 \
  --soil-moisture-vwc-percent 25 --soil-temperature-c 27 \
  --vegetation-coverage-percent 35 --vegetation-height-cm 25
```

## Sensor robustness evaluation

The robustness script evaluates Gaussian, impulse, drift, and freezing corruption from 5% through 30%. Noise is applied to direct numeric sensor channels; categorical context, the temporal `Hour` feature, and generator-derived `Inundation_Duration_Days` remain unchanged.

```bash
python src/evaluate_sensor_robustness.py --dataset data/philippines_rice_paddy_awd_dataset.csv
```

Results are exported to `data/sensor_robustness_results.csv`.

## Chapter 3 experiments

Run the reproducible Chapter 3 evaluation with a chronological 80/20 holdout and ten repeated corruption trials per sensor-fault condition:

```bash
python src/chapter3_experiments.py --repeats 10
```

The script writes the held-out baseline, dataset summaries, repeated robustness metrics, one-hour AWD-boundary projections, assumptions, and figures to `reports/chapter3_experiments/`, including `feature_fault_scope_safety.csv` for single-channel fault-routing scenarios. These scenarios flag one channel while leaving its values nominal, isolating controller routing rather than simulating sensor-error magnitude. For valid readings, the explicit rule used to generate the synthetic AWD labels supplies the controller action; the Random Forest is advisory, and a one-hour interlock enforces the configured simulated water-depth limits. The controller treats water depth as critical at every stage and soil moisture as critical outside flowering; faults in other measurement channels permit degraded policy operation. Because the policy and labels share the same rule, their agreement is not independent validation. The one-hour actuator rates and AWD limits are simulation assumptions, not field-calibrated measurements. Fault-channel flags are oracle inputs used to evaluate conditional behavior, not a validated real sensor-fault detector. The hold-last-command response for critical faults is not established as a safe field fallback. The controller simulation is software-only and does not constitute ESP32/Wokwi hardware-in-the-loop validation.

### ESP32/Wokwi controller demonstrator

The `wokwi/` directory contains a PlatformIO ESP32 firmware demonstrator for input validation, safe mode, and a one-hour AWD interlock. It receives test readings and an externally supplied requested action through the Wokwi serial monitor and displays gate and safe-mode states on LEDs. This standalone firmware does not yet implement the Python simulator's rule-driven command policy, and it does not embed the Random Forest. Follow `wokwi/README.md` to build and run it. A successful firmware build does not by itself mean a Wokwi simulation or physical hardware test was performed.

### Recall-prioritized model selection

For a separate, locked evaluation, the tuning script trains on four generated random-seed realizations (42–45), selects class weight and probability threshold on seed 46 by positive-class F2, then evaluates once on untouched seed 47. Every realization is generated by the same synthetic rules; this estimates consistency under those rules and is not a substitute for field validation. The selected threshold is also held fixed during robustness trials. The tuned classifier is advisory in the Python control simulation; the generated-rule AWD policy supplies valid-input actions.

```bash
python -m src.tune_gate_model --repeats 10
```

Outputs are written to `reports/chapter3_tuning/`, leaving the original chronological experiment artifacts unchanged. They include validation selection curves, the locked-test comparison and predictions, a serialized tuned model, repeated corruption metrics, combined-fault safety projections, single-channel fault-routing results in `feature_fault_scope_safety.csv`, assumptions, and figures. The untuned balanced model at threshold 0.50 is included as a comparison. Do not change the selected configuration after reviewing locked-test results and continue to call these synthetic-simulation results.

To use the tuned model with the existing one-sample predictor, pass the threshold recorded in `reports/chapter3_tuning/tuning_assumptions.json`:

```bash
python src/predict_gate_decision.py --model-path reports/chapter3_tuning/gate_model_tuned.joblib \
  --threshold 0.51 --month Jan --day-of-week Monday --hour 12 \
  --crop-growth-stage Vegetative/Tillering --water-table-depth-cm -15 \
  --inundation-duration-days 0 --water-flow-velocity-ms 0.10 --water-ph 6.8 \
  --soil-moisture-vwc-percent 25 --soil-temperature-c 27 \
  --vegetation-coverage-percent 35 --vegetation-height-cm 25
```

Omitting `--threshold` preserves the saved estimator's usual `predict()` behavior.

Run the controller and corruption unit tests with:

```bash
python -m unittest tests.test_chapter3_experiments -v
```

## Research limitation

This is a synthetic supervised-learning benchmark. It can support controlled algorithm and robustness experiments, but real deployment claims require field sensor data, observed gate actions, snail observations, and epidemiological validation.

### External irrigation time-series audit

The supplied `data/external_sources/Smart irrigation control system data with soil moisture, flow meter and electrovalve relay/` files can be audited separately from the synthetic training pipeline:

```bash
python -m src.evaluate_external_irrigation_data
```

The audit writes aggregate stream-quality and relay/moisture co-observation summaries to `reports/external_irrigation/`. It does not train or evaluate the Random Forest because the external streams do not provide its required feature set, and channel meanings and provenance still require confirmation. Relay/moisture summaries are descriptive only, not causal or safety evidence. Keep this dataset separate from training unless its source, units, and field semantics are verified.
