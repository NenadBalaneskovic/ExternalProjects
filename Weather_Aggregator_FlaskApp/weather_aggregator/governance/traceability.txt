"""
traceability.py
Model versioning and metadata tracking for Weather Aggregator.
Ensures reproducibility and auditability of forecasts.
"""

import hashlib
import json
from datetime import datetime

class Traceability:
    def __init__(self):
        self.registry = {}

    def register_model(self, model_name, params):
        """Register model with parameters and version hash."""
        version_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        metadata = {
            "model_name": model_name,
            "params": params,
            "version": version_hash,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.registry[model_name] = metadata
        return metadata

    def get_metadata(self, model_name):
        """Retrieve metadata for a registered model."""
        return self.registry.get(model_name, None)

    def list_registry(self):
        """List all registered models and metadata."""
        return self.registry
