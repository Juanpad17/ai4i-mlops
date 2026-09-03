"""Gate de decisión para retraining.

Este módulo integra la lógica de drift + performance para decidir si se
lanza el retraining del modelo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from src.monitoring.retraining_policy import should_retrain

DRIFT_SUMMARY_PATH = Path("reports/monitoring/data_drift_summary.json")
MODEL_SUMMARY_PATH = Path("reports/monitoring/model_monitoring_summary.json")
RETRAINING_DECISION_PATH = Path("reports/monitoring/retraining_decision.json")


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    drift_summary = load_json(DRIFT_SUMMARY_PATH)
    model_summary = load_json(MODEL_SUMMARY_PATH)

    decision = should_retrain(drift_summary, model_summary)

    output = {
        "decision": decision,
        "status": "RETRAINING_TRIGGERED" if decision["trigger_retraining"] else "NO_RETRAINING",
        "summary": {
            "drift_summary_path": str(DRIFT_SUMMARY_PATH),
            "model_summary_path": str(MODEL_SUMMARY_PATH),
        },
    }

    RETRAINING_DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRAINING_DECISION_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== GATE DE RETRAINING ===")
    print(f"Trigger retraining: {decision['trigger_retraining']}")
    print(f"Razón: {decision['reason']}")
    print(f"PSI promedio: {decision['psi_promedio']}")
    print(f"Variables con drift: {decision['variables_con_drift']}")
    print(f"Métricas: {decision.get('metrics', {'f1': None, 'recall': None, 'false_positive_rate': None})}")
    print(f"\nDecisión guardada en: {RETRAINING_DECISION_PATH}")


if __name__ == "__main__":
    main()
