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

The robustness script evaluates Gaussian, impulse, drift, and freezing corruption from 5% through 30%. Noise is applied to numeric sensor features while categorical context remains unchanged.

```bash
python src/evaluate_sensor_robustness.py --dataset data/philippines_rice_paddy_awd_dataset.csv
```

Results are exported to `data/sensor_robustness_results.csv`.

## Research limitation

This is a synthetic supervised-learning benchmark. It can support controlled algorithm and robustness experiments, but real deployment claims require field sensor data, observed gate actions, snail observations, and epidemiological validation.
