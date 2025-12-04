from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.model_functional import FunctionalModel


@model_metric(name="brier_score")
def brier_score(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: FunctionalModel,
) -> List[Measure]:
    """
    Brier Score = mean squared error between predicted probabilities and true labels
    Lower is better. Works for binary and multi-class classification
    """

    # 1) Extract feature columns X and target y
    target_col = datashape.target.name
    feature_cols = [f.name for f in datashape.features]

    df: pd.DataFrame = dataset.data
    if not isinstance(df, pd.DataFrame):
        raise TypeError("dataset.data is expected to be a pandas.DataFrame")

    X = df[feature_cols]
    y_true = df[target_col].to_numpy()

    n = len(X)
    if n == 0:
        return [Measure(name="brier_score", score=float("nan"), time=datetime.now())]

    batch_size = 10_000
    total_loss = 0.0
    total = 0

    # 2) Compute in batches
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        Xb_df = X.iloc[start:stop]

        # DataFrame -> torch tensor (float32) for torch models
        Xb = torch.from_numpy(Xb_df.to_numpy(dtype=np.float32))

        # 3) Get probabilities
        # Prefer predict_proba if available
        try:
            y_proba = functional_model.predict_proba(Xb)
        except (AttributeError, TypeError):
            try:
                y_proba = functional_model.predict_proba(model, Xb)
            except Exception:
                # Fallback to predict if predict_proba isn't there
                try:
                    y_proba = functional_model.predict(Xb)
                except TypeError:
                    y_proba = functional_model.predict(model, Xb)

        y_proba = np.asarray(y_proba)

        y_true_b = y_true[start:stop]

        # 4) Binary case
        if y_proba.ndim == 1:
            p = y_proba.astype(float)
            yb = y_true_b.astype(float)
            batch_loss = np.mean((p - yb) ** 2)

        # 5) Multi-class case
        else:
            # If probs are (n, C), build one-hot labels
            C = y_proba.shape[1]

            # Map labels to 0..C-1 if needed
            # (safe for weird label types)
            uniques = pd.unique(y_true_b)
            label_map = {lab: i for i, lab in enumerate(sorted(uniques))}
            y_idx = np.array([label_map[lab] for lab in y_true_b])

            Y_onehot = np.zeros_like(y_proba)
            Y_onehot[np.arange(len(y_idx)), y_idx] = 1

            batch_loss = np.mean((y_proba - Y_onehot) ** 2)

        total_loss += batch_loss * (stop - start)
        total += (stop - start)

    final_score = total_loss / total
    return [Measure(name="brier_score", score=float(final_score), time=datetime.now())]