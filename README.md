# Schisto-AI-Training

This repository contains a synthetic gate-decision prototype for an agricultural water-management robustness study.

## Final schema

The project uses the following columns in the labeled dataset:

- `water_level`
- `soil_moisture`
- `temperature`
- `water_velocity`
- `gate_state`
- `gate_decision`
- `previous_gate_decision`

`gate_decision` is the binary target variable:
- `1` = gate open
- `0` = gate close

The label is created from a transparent rule-based ground truth so the project remains defensible without a real field dataset with direct gate labels.

## Deterministic AWD + schistosomiasis gate state machine

I updated the synthetic ground-truth logic in [src/generate_synthetic_gate_dataset.py](src/generate_synthetic_gate_dataset.py) to implement a deterministic state machine based on the Philippine rice production AWD protocol and schistosomiasis-risk control.

The new rule-based function `determine_gate_state(...)` includes the requested IRRI/PRiSM constants:

- `WATER_LEVEL_TRIGGER_AWD = -15.0 cm`
- `WATER_LEVEL_TARGET_FLOOD = 5.0 cm`
- `VWC_SATURATED = 0.52`
- `VWC_AWD_TRIGGER = 0.30`
- `VWC_CRITICAL_WILTING = 0.18`
- `MIN_FLUSH_VELOCITY = 0.05 m/s`

The logic is organized as follows:

1. Dry-down control: if the previous gate state was `CLOSED`, the gate remains closed until the AWD trigger is reached (`water_level <= -15 cm` or `soil_moisture <= 0.30`).
2. Re-flooding hysteresis: once a trigger occurs, the gate opens and stays open until the flood ceiling is restored (`water_level >= 5 cm` and `soil_moisture >= 0.52`).
3. Snail habitat mitigation: when the gate is already open but the paddies are shallow and stagnant (`0 <= water_level <= 3 cm`, `water_velocity < 0.05 m/s`, warm conditions), the gate remains open to flush water and disrupt snail breeding micro-habitats.
4. Soil-moisture guardrail: if the water-level sensor is noisy but soil moisture falls below the wilting threshold (`<= 0.18`), the gate is forced open to protect the crop.

This creates a consistent, interpretable synthetic target that can still be used for model training or robustness analysis under noise.

## Scripts

- `src/generate_synthetic_gate_dataset.py`  
  Generates a synthetic labeled dataset with the final feature schema and rule-derived target.

- `src/train_gate_model.py`  
  Trains the final Random Forest classifier and saves the trained model and metrics.

- `src/predict_gate_decision.py`  
  Classifies a single new sensor sample and returns the predicted label and probability.

- `src/evaluate_sensor_robustness.py`  
  Tests the model under Gaussian noise, impulsive noise, sensor drift, and sensor freezing.

- `src/evaluate_minimal_test_csv.py`  
  Builds a minimal unlabeled CSV, predicts each row, and generates an HTML accuracy/precision/recall/F1 report.

## Generate the dataset

```bash
python src/generate_synthetic_gate_dataset.py --output data/synthetic_gate_dataset.csv
```

## Train the model

```bash
python src/train_gate_model.py --dataset data/synthetic_gate_dataset.csv --output-dir models
```

## Predict a new sample

```bash
python src/predict_gate_decision.py \
  --model-path models/gate_model_random_forest.joblib \
  --water-level 0.14 \
  --soil-moisture 0.31 \
  --temperature 27.5 \
  --water-velocity 0.09
```

## Evaluate model robustness

```bash
python src/evaluate_sensor_robustness.py --dataset data/synthetic_gate_dataset.csv
```

## Evaluate a small unlabeled CSV and create an HTML report

```bash
python src/evaluate_minimal_test_csv.py --model-path models/gate_model_random_forest.joblib
```

The generated dataset is stored in `data/synthetic_gate_dataset.csv` and the final model artifact is saved in `models/gate_model_random_forest.joblib`.
