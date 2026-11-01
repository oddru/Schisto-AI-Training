# Isolated soil-moisture forecasting experiment

This optional experiment evaluates a one-day-ahead soil-moisture forecast on
the public `it4lia/soil_moisture_dataset`. It follows the separate XGBoost
forecasting track in `it4lia/predictive_irrigation_models` without importing
that repository's pipeline or changing the project's Random Forest, controller,
synthetic datasets, or Chapter 3 reports.

The experiment uses only `Water Content` sensor streams from consortiums 0 and
2, where the dataset card says those sensors are available. It constructs
one-, two-, and seven-day lag features only when the corresponding observations
exist at exactly those calendar offsets. Each consortium is split separately
by target date (60/20/20); the XGBoost model for that cohort is fitted on its
first 80% after the split is fixed. It is compared with persistence (last
available daily reading) and a seven-day-naive baseline. Results are reported
per consortium and pooled over their chronological test records. Metrics
describe this forecasting task only and are not gate-action or AWD safety
results. Pooled R-squared is omitted because level differences between
consortiums can inflate it.

## Run

The experiment uses a dedicated virtual environment, so installing its
optional dependencies does not change the main project environment:

```powershell
python -m venv experiments\irrigation_forecasting\.venv
experiments\irrigation_forecasting\.venv\Scripts\python.exe -m pip install -r experiments\irrigation_forecasting\requirements.txt
experiments\irrigation_forecasting\.venv\Scripts\python.exe -m experiments.irrigation_forecasting.download_data
experiments\irrigation_forecasting\.venv\Scripts\python.exe -m experiments.irrigation_forecasting.evaluate
```

By default, the two relevant Parquet files are downloaded into this
experiment's ignored `data/` directory, and aggregate outputs are written to
its ignored `outputs/` directory. They can be regenerated or redirected:

```powershell
experiments\irrigation_forecasting\.venv\Scripts\python.exe -m experiments.irrigation_forecasting.evaluate `
  --data-dir experiments\irrigation_forecasting\data `
  --output-dir experiments\irrigation_forecasting\outputs
```

The script does not save model weights or copy/transform source files. Keep
downloaded data and generated outputs local. The dataset card lists a
CC BY-NC-ND 4.0 license; verify its terms and obtain appropriate guidance
before redistributing data, model artifacts, or derived results, including
thesis or public-report results.

## Scope and attribution

Dataset: `it4lia/soil_moisture_dataset`, produced by Fondazione Bruno Kessler
(FBK) OpenIoT research unit, Trentino, Italy. Dataset card:
<https://huggingface.co/datasets/it4lia/soil_moisture_dataset>.

Forecasting-method reference:
<https://huggingface.co/it4lia/predictive_irrigation_models>.
The reference configuration lists 500 estimators, depth 5, and learning rate
0.1; this experiment uses those settings and an absolute-error objective.

This is an external, non-Philippine forecasting benchmark. It does not include
the current model's `AWD_Gate_Action` target and must not be presented as
validation of the gate classifier, controller, AWD policy, or field safety.
