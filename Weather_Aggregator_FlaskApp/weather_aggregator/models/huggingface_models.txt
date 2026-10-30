"""
huggingface_models.py
Hosted model inference via Hugging Face API.
"""

import requests
from governance.config import Config
from governance.logger import get_logger

logger = get_logger(__name__)

def query_hf_model(model_name, payload):
    """Query Hugging Face hosted model."""
    headers = {"Authorization": f"Bearer {Config.HF_API_TOKEN}"}
    response = requests.post(Config.HF_API_URL + model_name, headers=headers, json=payload)
    if response.status_code == 200:
        logger.info(f"Model {model_name} inference successful.")
        return response.json()
    else:
        logger.error(f"Error {response.status_code} from {model_name}: {response.text}")
        return None

def batch_query(models, payload):
    """Query multiple Hugging Face models sequentially."""
    results = {}
    for model in models:
        results[model] = query_hf_model(model, payload)
    return results
