"""
__init__.py
Initialization for Weather Aggregator models package.
Exposes base models, hosted Hugging Face models, meta learner, and explainability utilities.
"""

from .base_models import (
    sarimax_forecast,
    kalman_filter_forecast,
    random_forest_forecast,
    gradient_boosting_forecast,
    lstm_forecast,
    cnn_forecast,
    autoencoder_forecast,
)

from .huggingface_models import query_hf_model, batch_query
from .meta_learner import MetaLearner
from .explainability import Explainability

__all__ = [
    # Base models
    "sarimax_forecast",
    "kalman_filter_forecast",
    "random_forest_forecast",
    "gradient_boosting_forecast",
    "lstm_forecast",
    "cnn_forecast",
    "autoencoder_forecast",
    # Hugging Face integration
    "query_hf_model",
    "batch_query",
    # Meta learner
    "MetaLearner",
    # Explainability
    "Explainability",
]
