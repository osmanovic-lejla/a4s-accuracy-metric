import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss

def main():
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVC (probabilities)": SVC(probability=True),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Predict probabilities for the positive class
        y_proba = model.predict_proba(X_test)[:, 1]

        # Compute Brier Score
        score = brier_score_loss(y_test, y_proba)
        results[name] = score

    print("\n=== BRIER SCORE RESULTS (lower is better) ===")
    for name, score in results.items():
        print(f"{name}: {score:.5f}")


if __name__ == "__main__":
    main()
