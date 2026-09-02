from src.monitoring.retraining_policy import should_retrain


def test_should_retrain_when_drift_is_severe_and_performance_degrades():
    drift_summary = {
        "psi_promedio": 0.31,
        "variables_con_drift": ["temperature_difference", "mechanical_power"],
    }
    model_summary = {
        "performance": {
            "f1": 0.72,
            "recall": 0.88,
            "false_positive_rate": 0.18,
        }
    }

    decision = should_retrain(drift_summary, model_summary)

    assert decision["trigger_retraining"] is True
    assert decision["reason"]


def test_should_not_retrain_when_drift_exists_but_performance_is_stable():
    drift_summary = {
        "psi_promedio": 0.18,
        "variables_con_drift": ["temperature_difference"],
    }
    model_summary = {
        "performance": {
            "f1": 0.92,
            "recall": 0.96,
            "false_positive_rate": 0.05,
        }
    }

    decision = should_retrain(drift_summary, model_summary)

    assert decision["trigger_retraining"] is False


def test_should_not_retrain_when_model_performance_not_available():
    drift_summary = {
        "psi_promedio": 0.40,
        "variables_con_drift": ["temperature_difference", "mechanical_power"],
    }
    model_summary = {}

    decision = should_retrain(drift_summary, model_summary)

    assert decision["trigger_retraining"] is False
