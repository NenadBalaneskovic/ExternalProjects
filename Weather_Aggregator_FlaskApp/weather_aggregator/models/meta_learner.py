"""
meta_learner.py
Meta learner that aggregates forecasts from base models.
Supports multiple methods (ridge, logistic, boosting).
"""

import logging
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


class MetaLearner:
    def __init__(self, method="ridge"):
        """
        Initialize the meta learner with a chosen method.

        Parameters
        ----------
        method : str
            Aggregation method. Options: "ridge", "logistic", "boosting".
        """
        if method == "ridge":
            self.model = Ridge()
        elif method == "logistic":
            self.model = LogisticRegression()
        elif method == "boosting":
            self.model = GradientBoostingRegressor()
        else:
            raise ValueError(f"Unknown method: {method}")

        self.method = method
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the meta learner on model outputs.

        Parameters
        ----------
        X : list or np.ndarray
            Shape (n_samples, n_models). Each row = predictions from base models.
        y : list or np.ndarray
            Shape (n_samples,). Target values (e.g., average of base predictions).
        """
        X = np.array(X)
        y = np.array(y)

        logger.info(f"MetaLearner.fit called with X shape={X.shape}, y shape={y.shape}")

        # Guard clause: check consistency
        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.warning("Empty input to MetaLearner.fit. Skipping training.")
            self.is_fitted = False
            return

        if X.shape[0] != y.shape[0]:
            logger.error(
                f"Inconsistent samples: X has {X.shape[0]} rows, y has {y.shape[0]} rows."
            )
            # Fallback: recompute y as row averages
            y = np.array([np.mean(row) for row in X])
            logger.info(f"Recomputed y from X. New y shape={y.shape}")

        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info(f"MetaLearner ({self.method}) successfully fitted.")
        except Exception as e:
            logger.exception(f"MetaLearner.fit failed: {e}")
            self.is_fitted = False

    def predict(self, X):
        """
        Predict ensemble forecast.

        Parameters
        ----------
        X : list or np.ndarray
            Shape (n_samples, n_models). Each row = predictions from base models.

        Returns
        -------
        np.ndarray
            Ensemble predictions.
        """
        X = np.array(X)
        logger.info(f"MetaLearner.predict called with X shape={X.shape}")

        if not self.is_fitted:
            logger.warning("MetaLearner not fitted. Returning row averages as fallback.")
            return np.array([np.mean(row) for row in X])

        try:
            preds = self.model.predict(X)
            logger.info("MetaLearner.predict succeeded.")
            return preds
        except Exception as e:
            logger.exception(f"MetaLearner.predict failed: {e}")
            return np.array([np.mean(row) for row in X])
