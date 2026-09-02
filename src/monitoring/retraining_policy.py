"""Regla de decisión para disparar retraining en producción.

La lógica sigue la recomendación de MLOps: no basta con detectar drift en
la distribución de entrada; también debe confirmarse que el modelo tuvo una
pérdida real de performance antes de ejecutar un retraining automático.
"""

from __future__ import annotations

from typing import Any, Dict


DEFAULT_THRESHOLDS = {
    "psi_severe": 0.25,
    "min_drift_variables": 2,
    "f1_threshold": 0.80,
    "recall_threshold": 0.90,
    "false_positive_rate_threshold": 0.15,
}


def _get_metric(model_summary: Dict[str, Any], metric_name: str, default: float | None = None) -> float | None:
    """Obtiene una métrica desde el resumen del modelo, soportando estructura anidada."""
    performance = model_summary.get("performance", {}) if isinstance(model_summary, dict) else {}
    if isinstance(performance, dict):
        value = performance.get(metric_name)
        if value is not None:
            return float(value)

    value = model_summary.get(metric_name)
    if value is not None:
        return float(value)

    return default


def should_retrain(
    drift_summary: Dict[str, Any] | None,
    model_summary: Dict[str, Any] | None,
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Evalúa si debe dispararse el retraining.

    Regla de decisión:
      - Drift severo en la distribución de entrada
      - y degradación real de métricas del modelo
      => retraining.

    El retorno incluye la decisión y una explicación legible.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS

    if drift_summary is None:
        drift_summary = {}
    if model_summary is None:
        model_summary = {}

    psi_promedio = float(drift_summary.get("psi_promedio", 0.0) or 0.0)
    variables_con_drift = drift_summary.get("variables_con_drift", []) or []
    variables_con_drift = list(variables_con_drift)

    drift_severo = (
        psi_promedio > thresholds["psi_severe"]
        or len(variables_con_drift) >= thresholds["min_drift_variables"]
    )

    f1 = _get_metric(model_summary, "f1")
    recall = _get_metric(model_summary, "recall")
    false_positive_rate = _get_metric(model_summary, "false_positive_rate")

    metrics_available = any(
        metric is not None for metric in (f1, recall, false_positive_rate)
    )

    if not metrics_available:
        return {
            "trigger_retraining": False,
            "reason": "No hay métricas de performance disponibles para confirmar degradación.",
            "drift_severo": drift_severo,
            "performance_degraded": False,
            "psi_promedio": psi_promedio,
            "variables_con_drift": variables_con_drift,
            "metrics": {
                "f1": None,
                "recall": None,
                "false_positive_rate": None,
            },
            "thresholds": thresholds,
        }

    performance_degraded = (
        (f1 is not None and f1 < thresholds["f1_threshold"])
        or (recall is not None and recall < thresholds["recall_threshold"])
        or (
            false_positive_rate is not None
            and false_positive_rate > thresholds["false_positive_rate_threshold"]
        )
    )

    trigger_retraining = bool(drift_severo and performance_degraded)

    if trigger_retraining:
        reason = (
            "Se detectó drift severo en la distribución de entrada y, simultáneamente, "
            "la performance del modelo cayó por debajo del umbral aceptable. "
            "Por lo tanto, se recomienda disparar el retraining."
        )
    else:
        reason = (
            "No se cumple la condición de retraining: el drift no fue severo o la "
            "performance del modelo sigue dentro del rango aceptable."
        )

    return {
        "trigger_retraining": trigger_retraining,
        "reason": reason,
        "drift_severo": drift_severo,
        "performance_degraded": performance_degraded,
        "psi_promedio": psi_promedio,
        "variables_con_drift": variables_con_drift,
        "metrics": {
            "f1": f1,
            "recall": recall,
            "false_positive_rate": false_positive_rate,
        },
        "thresholds": thresholds,
    }
