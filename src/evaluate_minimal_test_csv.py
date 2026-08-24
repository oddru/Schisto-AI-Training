import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from generate_synthetic_gate_dataset import compute_risk_score

FEATURES = [
    "water_level",
    "soil_moisture",
    "water_temp_c",
    "turbidity_ntu",
    "rain_mm",
]


def build_minimal_test_csv(output_path: str = "data/minimal_gate_test.csv") -> pd.DataFrame:
    rows = [
        {"water_level": 0.06, "soil_moisture_top": 0.45, "soil_moisture_mid": 0.42, "soil_moisture_deep": 0.39, "groundwater_depth": 0.22, "water_temp_c": 29.5, "turbidity_ntu": 42.0, "rain_mm": 8.0, "flow_rate": 0.08, "ec_us": 240.0, "ndvi": 0.45, "et_mm": 4.0, "crop_stage": 1},
        {"water_level": 0.12, "soil_moisture_top": 0.36, "soil_moisture_mid": 0.35, "soil_moisture_deep": 0.33, "groundwater_depth": 0.18, "water_temp_c": 27.0, "turbidity_ntu": 55.0, "rain_mm": 11.0, "flow_rate": 0.10, "ec_us": 260.0, "ndvi": 0.38, "et_mm": 3.2, "crop_stage": 1},
        {"water_level": 0.28, "soil_moisture_top": 0.22, "soil_moisture_mid": 0.20, "soil_moisture_deep": 0.18, "groundwater_depth": 0.30, "water_temp_c": 21.0, "turbidity_ntu": 75.0, "rain_mm": 18.0, "flow_rate": 0.24, "ec_us": 350.0, "ndvi": 0.28, "et_mm": 2.1, "crop_stage": 0},
        {"water_level": 0.33, "soil_moisture_top": 0.19, "soil_moisture_mid": 0.18, "soil_moisture_deep": 0.17, "groundwater_depth": 0.35, "water_temp_c": 20.0, "turbidity_ntu": 90.0, "rain_mm": 16.0, "flow_rate": 0.30, "ec_us": 500.0, "ndvi": 0.20, "et_mm": 1.8, "crop_stage": 0},
        {"water_level": 0.09, "soil_moisture_top": 0.50, "soil_moisture_mid": 0.48, "soil_moisture_deep": 0.44, "groundwater_depth": 0.20, "water_temp_c": 31.0, "turbidity_ntu": 35.0, "rain_mm": 4.0, "flow_rate": 0.09, "ec_us": 180.0, "ndvi": 0.52, "et_mm": 5.0, "crop_stage": 2},
        {"water_level": 0.2, "soil_moisture_top": 0.31, "soil_moisture_mid": 0.30, "soil_moisture_deep": 0.28, "groundwater_depth": 0.16, "water_temp_c": 25.0, "turbidity_ntu": 68.0, "rain_mm": 13.0, "flow_rate": 0.18, "ec_us": 290.0, "ndvi": 0.33, "et_mm": 3.8, "crop_stage": 1},
        {"water_level": 0.05, "soil_moisture_top": 0.53, "soil_moisture_mid": 0.50, "soil_moisture_deep": 0.46, "groundwater_depth": 0.40, "water_temp_c": 30.0, "turbidity_ntu": 28.0, "rain_mm": 3.0, "flow_rate": 0.06, "ec_us": 170.0, "ndvi": 0.55, "et_mm": 5.2, "crop_stage": 1},
        {"water_level": 0.40, "soil_moisture_top": 0.18, "soil_moisture_mid": 0.15, "soil_moisture_deep": 0.12, "groundwater_depth": 0.10, "water_temp_c": 19.0, "turbidity_ntu": 100.0, "rain_mm": 24.0, "flow_rate": 0.32, "ec_us": 420.0, "ndvi": 0.22, "et_mm": 1.6, "crop_stage": 3},
        {"water_level": 0.15, "soil_moisture_top": 0.35, "soil_moisture_mid": 0.33, "soil_moisture_deep": 0.31, "groundwater_depth": 0.20, "water_temp_c": 26.5, "turbidity_ntu": 50.0, "rain_mm": 12.0, "flow_rate": 0.12, "ec_us": 250.0, "ndvi": 0.36, "et_mm": 3.5, "crop_stage": 1},
        {"water_level": 0.11, "soil_moisture_top": 0.47, "soil_moisture_mid": 0.44, "soil_moisture_deep": 0.40, "groundwater_depth": 0.22, "water_temp_c": 28.0, "turbidity_ntu": 46.0, "rain_mm": 7.5, "flow_rate": 0.11, "ec_us": 215.0, "ndvi": 0.49, "et_mm": 4.6, "crop_stage": 2},
        {"water_level": 0.52, "soil_moisture_top": 0.14, "soil_moisture_mid": 0.12, "soil_moisture_deep": 0.10, "groundwater_depth": 0.08, "water_temp_c": 18.0, "turbidity_ntu": 110.0, "rain_mm": 27.0, "flow_rate": 0.45, "ec_us": 600.0, "ndvi": 0.12, "et_mm": 1.2, "crop_stage": 3},
        {"water_level": 0.07, "soil_moisture_top": 0.49, "soil_moisture_mid": 0.47, "soil_moisture_deep": 0.44, "groundwater_depth": 0.24, "water_temp_c": 30.5, "turbidity_ntu": 40.0, "rain_mm": 5.0, "flow_rate": 0.07, "ec_us": 195.0, "ndvi": 0.50, "et_mm": 4.9, "crop_stage": 1},
    ]

    df = pd.DataFrame(rows)
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def compute_ground_truth(df: pd.DataFrame) -> pd.Series:
    risk = compute_risk_score(df)
    return (risk > 4.5).astype(int)


def evaluate_model(model_path: str, data_path: str, report_path: str) -> dict:
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    y_true = compute_ground_truth(df)
    X = df[FEATURES]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    df["ground_truth"] = y_true.values
    df["predicted_gate"] = y_pred
    df["probability_gate_open"] = y_prob
    df["correct_prediction"] = (y_pred == y_true.values)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }

    report_dir = Path(report_path).parent
    report_dir.mkdir(parents=True, exist_ok=True)

    html = render_html_report(df, metrics)
    Path(report_path).write_text(html, encoding="utf-8")
    return {"metrics": metrics, "df": df}


def render_html_report(df: pd.DataFrame, metrics: dict) -> str:
    rows_html = []
    for _, row in df.iterrows():
        rows_html.append(
            """
            <tr>
                <td>{row_id}</td>
                <td>{water_level}</td>
                    <td>{soil_moisture_top}</td>
                    <td>{soil_moisture_mid}</td>
                    <td>{soil_moisture_deep}</td>
                    <td>{groundwater_depth}</td>
                    <td>{water_temp_c}</td>
                    <td>{turbidity_ntu}</td>
                    <td>{rain_mm}</td>
                    <td>{ndvi:.3f}</td>
                    <td>{et_mm:.2f}</td>
                    <td>{ground_truth}</td>
                    <td>{predicted_gate}</td>
                    <td>{probability_gate_open:.4f}</td>
                    <td>{correct_prediction}</td>
                </tr>
                """.format(
                    row_id=int(_),
                    water_level=row["water_level"],
                    soil_moisture_top=row["soil_moisture_top"],
                    soil_moisture_mid=row["soil_moisture_mid"],
                    soil_moisture_deep=row["soil_moisture_deep"],
                    groundwater_depth=row["groundwater_depth"],
                    water_temp_c=row["water_temp_c"],
                    turbidity_ntu=row["turbidity_ntu"],
                    rain_mm=row["rain_mm"],
                    ndvi=row["ndvi"],
                    et_mm=row["et_mm"],
                    ground_truth=int(row["ground_truth"]),
                    predicted_gate=int(row["predicted_gate"]),
                    probability_gate_open=float(row["probability_gate_open"]),
                    correct_prediction=str(row["correct_prediction"]),
                )
            )

    metrics_html = f"""
    <div class="metrics">
        <div class="metric"><span>Accuracy</span><strong>{metrics['accuracy']:.4f}</strong></div>
        <div class="metric"><span>Precision</span><strong>{metrics['precision']:.4f}</strong></div>
        <div class="metric"><span>Recall</span><strong>{metrics['recall']:.4f}</strong></div>
        <div class="metric"><span>F1</span><strong>{metrics['f1']:.4f}</strong></div>
    </div>
    """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Gate Model Evaluation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 32px; background: #f5f7fb; color: #1d2433; }}
            h1, h2 {{ color: #17324d; }}
            .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 16px; margin: 24px 0; }}
            .metric {{ background: #fff; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
            .metric span {{ display: block; font-size: 12px; color: #58657a; margin-bottom: 8px; text-transform: uppercase; }}
            .metric strong {{ font-size: 22px; }}
            table {{ width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
            th, td {{ border: 1px solid #dfe7f2; padding: 10px; text-align: center; font-size: 13px; }}
            th {{ background: #eaf1fb; }}
            .correct {{ color: green; font-weight: bold; }}
            .wrong {{ color: #b42318; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Gate Model Evaluation Report</h1>
        <p>This report evaluates the trained model on a new CSV containing no target label column. The actual label is generated from the same domain rule used to create the synthetic dataset, purely for evaluation.</p>
        {metrics_html}
        <h2>Prediction Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Row</th>
                            <th>Water Level (m)</th>
                            <th>Soil Moisture</th>
                            <th>Temp (C)</th>
                            <th>Turbidity</th>
                            <th>Rain</th>
                    <th>Ground Truth</th>
                    <th>Predicted Gate</th>
                    <th>Gate Open Prob.</th>
                    <th>Correct?</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </body>
    </html>
    """


if __name__ == "__main__":
    csv_path = "data/minimal_gate_test.csv"
    report_path = "reports/model_evaluation_report.html"
    model_path = "models/gate_model_random_forest.joblib"

    build_minimal_test_csv(csv_path)
    result = evaluate_model(model_path, csv_path, report_path)
    metrics = result["metrics"]
    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }, indent=2))
    print(f"Saved test CSV to: {csv_path}")
    print(f"Saved HTML report to: {report_path}")
