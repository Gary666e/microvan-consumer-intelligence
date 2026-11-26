"""Offline monitoring example. Thresholds are proposed review gates."""
from pathlib import Path
import pandas as pd
import json

def monitor(ref, cur):
    alerts = []
    for name, flag, action in [
        ("score drift", cur.score_psi >= .25, "Investigate segment mix and score drift."),
        ("recall degradation", ref.recall - cur.recall >= .10, "Inspect segment recall and coverage."),
        ("capacity utilization", cur.selection_rate < .8 * ref.selection_rate, "Review threshold and capacity."),
        ("AUC degradation", ref.auc - cur.auc >= .05, "Validate a challenger on independent labels.")
    ]:
        if flag:
            alerts.append({"alert": name, "action": action})
    return {"status": "REVIEW" if alerts else "PASS", "alerts": alerts,
            "release_gate": "Require independently labeled validation and owner approval."}

if __name__ == "__main__":
    data = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "monitoring_stress.csv")
    baseline = monitor(data.iloc[0], data.iloc[0])
    stress = monitor(data.iloc[0], data.iloc[1])
    assert baseline["status"] == "PASS"
    assert stress["status"] == "REVIEW"
    print(json.dumps({"baseline": baseline, "stress": stress}, indent=2))
