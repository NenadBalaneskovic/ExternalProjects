"""
explainability.py
Explainability utilities for Weather Aggregator.
Provides SHAP and LIME analysis for ensemble forecasts.
"""

import shap
from lime.lime_tabular import LimeTabularExplainer

class Explainability:
    def __init__(self, training_data, feature_names=None):
        self.training_data = training_data
        self.feature_names = feature_names if feature_names else [f"f{i}" for i in range(training_data.shape[1])]

    def shap_analysis(self, model, sample):
        explainer = shap.Explainer(model, self.training_data)
        shap_values = explainer(sample)
        return shap_values

    def lime_analysis(self, model, sample):
        explainer = LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            verbose=True,
            mode="regression"
        )
        explanation = explainer.explain_instance(sample, model.predict, num_features=5)
        return explanation
