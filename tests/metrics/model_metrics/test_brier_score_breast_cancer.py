import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss


def test_brier_score_on_real_models_breast_cancer():
    """
    Train multiple models on the Breast Cancer dataset,
    compute their Brier Scores, save results as CSV,
    and check basic validity.
    """

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    #  Define models 
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVC (probabilities)": SVC(probability=True),
    }

    results = []

    #  Train models & compute Brier Scores 
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        score = brier_score_loss(y_test, y_proba)

        # Store result in list for saving
        results.append({"model": name, "brier_score": float(score)})

        # Basic checks
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert 0 <= score <= 1, f"Invalid score for {name}: {score}"

    #  Ensure models are different 
    unique_scores = set(round(r["brier_score"], 5) for r in results)
    assert len(unique_scores) >= 2

    # Save results to CSV 
    csv_path = Path("tests/data/measures/brier_score_breast_cancer.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(csv_path, index=False)

    # Ensure file was created
    assert csv_path.exists()