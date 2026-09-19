"""Render the expanded-dataset robustness CSV as a simple HTML report."""

from pathlib import Path

import pandas as pd


def render(input_path: str, output_path: str) -> None:
    results = pd.read_csv(input_path)
    sections = []
    for noise_level, group in results.groupby("noise_level", sort=True):
        rows = []
        for _, row in group.iterrows():
            rows.append(
                "<tr>"
                f"<td>{row['fault_type']}</td>"
                f"<td>{row['accuracy']:.2%}</td>"
                f"<td>{row['precision']:.2%}</td>"
                f"<td>{row['recall']:.2%}</td>"
                f"<td>{row['f1_score']:.2%}</td>"
                f"<td>{row['mae']:.4f}</td>"
                "</tr>"
            )
        sections.append(
            f"<h2>Fault Type ({int(noise_level)}%)</h2>"
            "<table><thead><tr>"
            f"<th>Fault Type ({int(noise_level)}%)</th><th>Accuracy</th>"
            "<th>Precision</th><th>Recall</th><th>F1-Score</th><th>MAE</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>AWD Gate Robustness Evaluation</title>
<style>
body{background:#000;color:#bafecf;font-family:"Courier New",monospace;padding:20px}
main{max-width:1100px;margin:auto;border:1px solid #2efc7f;padding:18px;background:#06110c}
h1,h2{color:#7efc9d;text-transform:uppercase;letter-spacing:.08em}
table{width:100%;border-collapse:collapse;margin:0 0 24px}
th,td{border:1px solid #2efc7f;padding:8px;text-align:center}
th{background:#123f26;color:#d7ffe7} td{background:#0c2214}
</style></head><body><main>
<h1>AWD Gate Robustness Evaluation</h1>
<p>Random Forest trained on the expanded synthetic Philippine rice-paddy dataset.
Metrics are computed against the synthetic AWD_Gate_Action labels.</p>
""" + "".join(sections) + "</main></body></html>"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    render(
        "data/sensor_robustness_results.csv",
        "reports/model_evaluation_report.html",
    )
    print("Wrote reports/model_evaluation_report.html")
