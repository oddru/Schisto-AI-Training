import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from generate_synthetic_gate_dataset import compute_risk_score

FEATURES = [
    "flow_rate",
    "water_temp_c",
    "turbidity_ntu",
    "rain_mm",
    "conductivity_us",
]


def build_minimal_test_csv(output_path: str = "data/minimal_gate_test.csv") -> pd.DataFrame:
    rows = [
        {"flow_rate": 0.06, "water_temp_c": 29.5, "turbidity_ntu": 42.0, "rain_mm": 8.0, "conductivity_us": 210.0},
        {"flow_rate": 0.12, "water_temp_c": 27.0, "turbidity_ntu": 55.0, "rain_mm": 11.0, "conductivity_us": 260.0},
        {"flow_rate": 0.28, "water_temp_c": 21.0, "turbidity_ntu": 75.0, "rain_mm": 18.0, "conductivity_us": 350.0},
        {"flow_rate": 0.33, "water_temp_c": 20.0, "turbidity_ntu": 90.0, "rain_mm": 16.0, "conductivity_us": 500.0},
        {"flow_rate": 0.09, "water_temp_c": 31.0, "turbidity_ntu": 35.0, "rain_mm": 4.0, "conductivity_us": 180.0},
        {"flow_rate": 0.20, "water_temp_c": 25.0, "turbidity_ntu": 68.0, "rain_mm": 13.0, "conductivity_us": 290.0},
        {"flow_rate": 0.05, "water_temp_c": 30.0, "turbidity_ntu": 28.0, "rain_mm": 3.0, "conductivity_us": 170.0},
        {"flow_rate": 0.40, "water_temp_c": 19.0, "turbidity_ntu": 100.0, "rain_mm": 24.0, "conductivity_us": 420.0},
        {"flow_rate": 0.15, "water_temp_c": 26.5, "turbidity_ntu": 50.0, "rain_mm": 12.0, "conductivity_us": 250.0},
        {"flow_rate": 0.11, "water_temp_c": 28.0, "turbidity_ntu": 46.0, "rain_mm": 7.5, "conductivity_us": 215.0},
        {"flow_rate": 0.52, "water_temp_c": 18.0, "turbidity_ntu": 110.0, "rain_mm": 27.0, "conductivity_us": 600.0},
        {"flow_rate": 0.07, "water_temp_c": 30.5, "turbidity_ntu": 40.0, "rain_mm": 5.0, "conductivity_us": 195.0},
    ]

    df = pd.DataFrame(rows)
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def compute_ground_truth(df: pd.DataFrame) -> pd.Series:
    risk = compute_risk_score(df)
    return (risk > 5.0).astype(int)


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
                <td>{flow_rate}</td>
                <td>{water_temp_c}</td>
                <td>{turbidity_ntu}</td>
                <td>{rain_mm}</td>
                <td>{conductivity_us}</td>
                <td>{ground_truth}</td>
                <td>{predicted_gate}</td>
                <td>{probability_gate_open:.4f}</td>
                <td>{correct_prediction}</td>
            </tr>
            """.format(
                row_id=int(_),
                flow_rate=row["flow_rate"],
                water_temp_c=row["water_temp_c"],
                turbidity_ntu=row["turbidity_ntu"],
                rain_mm=row["rain_mm"],
                conductivity_us=row["conductivity_us"],
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
                    <th>Flow Rate</th>
                    <th>Temp (C)</th>
                    <th>Turbidity</th>
                    <th>Rain</th>
                    <th>Conductivity</th>
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
