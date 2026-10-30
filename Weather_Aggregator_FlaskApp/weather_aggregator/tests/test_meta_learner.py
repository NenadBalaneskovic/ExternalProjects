"""
test_meta_learner.py
Unit tests for meta learner aggregation.
"""

import numpy as np
from models.meta_learner import MetaLearner

def test_ridge_meta_learner():
    X = np.array([[1,2],[2,3],[3,4]])
    y = np.array([2,3,4])
    meta = MetaLearner(method="ridge")
    meta.fit(X, y)
    preds = meta.predict(X)
    assert len(preds) == len(y)

def test_confidence_band():
    forecasts = np.array([[1,2,3],[2,3,4],[3,4,5]])
    meta = MetaLearner(method="ridge")
    lower, upper = meta.confidence_band(forecasts)
    assert len(lower) == forecasts.shape[0]
    assert len(upper) == forecasts.shape[0]
