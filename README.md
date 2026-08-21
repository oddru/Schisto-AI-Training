# Schisto-AI-Training

This repository contains synthetic-data scripts for a schistosomiasis mitigation gate-control robustness study.

## Included scripts

- `src/generate_synthetic_gate_dataset.py`
  Generates a hydrological dataset with a rule-based `gate_open` label based on the environmental conditions used in the proposal.

- `src/evaluate_sensor_robustness.py`
  Trains a simple classifier and evaluates robustness under Gaussian noise, impulsive noise, sensor drift, and sensor freezing.

## Example usage

Generate the dataset:

```bash
python src/generate_synthetic_gate_dataset.py --output data/synthetic_gate_dataset.csv
```

Train the default random-forest model:

```bash
python src/train_gate_model.py --dataset data/synthetic_gate_dataset.csv --output-dir models --model random_forest
```

Train the logistic-regression baseline if needed:

```bash
python src/train_gate_model.py --dataset data/synthetic_gate_dataset.csv --output-dir models --model logistic_regression
```

Classify a new sample:

```bash
python src/predict_gate_decision.py \
  --model-path models/gate_model_random_forest.joblib \
  --flow-rate 0.08 \
  --water-temp-c 28.5 \
  --turbidity-ntu 45 \
  --rain-mm 9 \
  --conductivity-us 240
```

Evaluate robustness:

```bash
python src/evaluate_sensor_robustness.py --dataset data/synthetic_gate_dataset.csv
```

The generated dataset is stored in `data/synthetic_gate_dataset.csv`. Trained models are saved in the `models/` directory.