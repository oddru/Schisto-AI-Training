import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from generate_synthetic_gate_dataset import compute_risk_score

FEATURES = [
    "water_level",
    "soil_moisture",
    "temperature",
    "water_velocity",
]


def build_minimal_test_csv(output_path: str = "data/minimal_gate_test.csv") -> pd.DataFrame:
    rows = [
        {"water_level": 0.08, "soil_moisture": 0.28, "temperature": 30.0, "water_velocity": 0.05},
        {"water_level": 0.12, "soil_moisture": 0.32, "temperature": 28.5, "water_velocity": 0.08},
        {"water_level": 0.32, "soil_moisture": 0.18, "temperature": 20.0, "water_velocity": 0.25},
        {"water_level": 0.40, "soil_moisture": 0.14, "temperature": 19.5, "water_velocity": 0.35},
        {"water_level": 0.20, "soil_moisture": 0.50, "temperature": 31.0, "water_velocity": 0.10},
        {"water_level": 0.18, "soil_moisture": 0.39, "temperature": 25.0, "water_velocity": 0.14},
        {"water_level": 0.06, "soil_moisture": 0.52, "temperature": 30.5, "water_velocity": 0.04},
        {"water_level": 0.55, "soil_moisture": 0.12, "temperature": 18.5, "water_velocity": 0.45},
        {"water_level": 0.15, "soil_moisture": 0.36, "temperature": 26.5, "water_velocity": 0.12},
        {"water_level": 0.10, "soil_moisture": 0.48, "temperature": 29.0, "water_velocity": 0.09},
    ]

    df = pd.DataFrame(rows)
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def compute_ground_truth(df: pd.DataFrame) -> pd.Series:
    risk = compute_risk_score(df)
    return (risk > 2.2).astype(int)


def render_html_report(df: pd.DataFrame, metrics: dict) -> str:
    rows_html = []
    for idx, row in df.iterrows():
        rows_html.append(
            f"""
            <tr>
                <td>{idx}</td>
                <td>{row['water_level']:.3f}</td>
                <td>{row['soil_moisture']:.3f}</td>
                <td>{row['temperature']:.2f}</td>
                <td>{row['water_velocity']:.3f}</td>
                <td>{int(row['ground_truth'])}</td>
                <td>{int(row['predicted_gate'])}</td>
                <td>{float(row['probability_gate_open']):.4f}</td>
                <td>{str(row['correct_prediction'])}</td>
            </tr>
            """
        )

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Gate Decision Evaluation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 32px; background: #f5f7fb; color: #1f2937; }}
            h1, h2 {{ color: #0f172a; }}
            .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin: 20px 0; }}
            .metric {{ background: white; border-radius: 10px; padding: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }}
            th, td {{ border: 1px solid #dfe3eb; padding: 8px; text-align: center; }}
            th {{ background: #e2e8f0; }}
        </style>
    </head>
    <body>
        <h1>Gate Decision Evaluation</h1>
        <div class="metrics">
            <div class="metric"><strong>Accuracy</strong><br>{metrics['accuracy']:.4f}</div>
            <div class="metric"><strong>Precision</strong><br>{metrics['precision']:.4f}</div>
            <div class="metric"><strong>Recall</strong><br>{metrics['recall']:.4f}</div>
            <div class="metric"><strong>F1</strong><br>{metrics['f1']:.4f}</div>
        </div>

        <h2>Row-level predictions</h2>
        <table>
            <thead>
                <tr>
                    <th>Row</th>
                    <th>Water Level</th>
                    <th>Soil Moisture</th>
                    <th>Temperature</th>
                    <th>Water Velocity</th>
                    <th>Ground Truth</th>
                    <th>Predicted</th>
                    <th>Open Probability</th>
                    <th>Correct</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </body>
    </html>
    """


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
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    report_dir = Path(report_path).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(render_html_report(df, metrics), encoding="utf-8")
    return {"metrics": metrics, "df": df}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a minimal unlabeled CSV and evaluate a trained gate-decision model.")
    parser.add_argument("--model-path", type=str, default="models/gate_model_random_forest.joblib", help="Path to the saved model.")
    parser.add_argument("--input-csv", type=str, default="data/minimal_gate_test.csv", help="Unlabeled CSV to classify.")
    parser.add_argument("--report-path", type=str, default="reports/model_evaluation_report.html", help="Path to save the HTML report.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_minimal_test_csv(args.input_csv)
    result = evaluate_model(args.model_path, args.input_csv, args.report_path)
    print(json.dumps(result["metrics"], indent=2))
    print(f"HTML report written to {args.report_path}")
