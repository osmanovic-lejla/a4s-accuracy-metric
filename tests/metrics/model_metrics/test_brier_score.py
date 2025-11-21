from a4s_eval.metric_registries.model_metric_registry import model_metric_registry


def test_brier_score_is_registered():
    names = [name for name, _ in model_metric_registry]
    assert "brier_score" in names