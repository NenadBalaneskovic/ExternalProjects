# models/local_llm.py
from langchain.llms import HuggingFacePipeline
from transformers import pipeline


def load_local_llm():
    hf_pipeline = pipeline("text-generation",
                           model="mistralai/Mistral-7B-Instruct-v0.1",
                           max_new_tokens=256)
    return HuggingFacePipeline(pipeline=hf_pipeline)
