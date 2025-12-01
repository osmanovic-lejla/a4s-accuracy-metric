import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss


def test_brier_score_on_real_models_breast_cancer():
    """
    This test trains multiple sklearn models on the Breast Cancer dataset
    and verifies that Brier Score produces valid numerical results.
    """

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models we want to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVC (probabilities)": SVC(probability=True),
    }

    results = {}

    # Train each model and compute Brier Score
    for name, model in models.items():
        model.fit(X_train, y_train)

        # Probabilities for the positive class (label = 1)
        y_proba = model.predict_proba(X_test)[:, 1]

        score = brier_score_loss(y_test, y_proba)
        results[name] = score

    # Validate results
    for model_name, score in results.items():
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert 0.0 <= score <= 1.0, f"Brier score out of range: {model_name}"

    # Ensure not all scores are identical → metric actually distinguishes models
    unique_vals = set(round(v, 5) for v in results.values())
    assert len(unique_vals) >= 2
