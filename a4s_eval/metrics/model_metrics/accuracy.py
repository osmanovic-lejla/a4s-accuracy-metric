from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
import torch

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.model_functional import FunctionalModel


@model_metric(name="accuracy")
def accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: FunctionalModel,
) -> List[Measure]:
    """
    Classification accuracy = (# correct predictions) / (# total examples)
    Batches by 10k to avoid memory issues. Handles both label arrays and
    probability matrices (argmax). Converts DataFrames to Tensors for torch models.
    """

    # Extract X (features) and y (target) from the provided dataset using datashape
    target_col = datashape.target.name
    feature_cols = [f.name for f in datashape.features]

    df: pd.DataFrame = dataset.data
    if not isinstance(df, pd.DataFrame):
        raise TypeError("dataset.data is expected to be a pandas.DataFrame")

    X = df[feature_cols]
    y_true = df[target_col].to_numpy()

    n = len(X)
    if n == 0:
        return [Measure(name="accuracy", score=float("nan"), time=datetime.now())]

    batch_size = 10_000
    correct = 0
    total = 0

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        Xb_df = X.iloc[start:stop]

        # DataFrame -> float32 Tensor for PyTorch model
        Xb = torch.from_numpy(Xb_df.to_numpy(dtype=np.float32))

        # Call the functional model (supports both signatures)
        try:
            y_pred = functional_model.predict(Xb)
        except TypeError:
            y_pred = functional_model.predict(model, Xb)

        y_pred = np.asarray(y_pred)

        # If model returns probabilities/logits, take argmax
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(axis=1)

        # Match dtype for fair comparison
        y_true_b = y_true[start:stop]
        if y_pred.dtype != y_true_b.dtype:
            try:
                y_pred = y_pred.astype(y_true_b.dtype)
            except Exception:
                y_pred = y_pred.astype(str)
                y_true_b = y_true_b.astype(str)

        correct += (y_pred == y_true_b).sum()
        total += (stop - start)

    acc = correct / total if total else float("nan")
    return [Measure(name="accuracy", score=float(acc), time=datetime.now())]