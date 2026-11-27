# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  core/router.py
ROLE:    Semantic routing engine analyzing prompt payloads to intelligently
         distribute computational loads across the 4 free API-key tiers.
================================================================================
"""

import logging
from core.database import SessionPersistenceManager

log = logging.getLogger("SemanticRouter")


class IntentPreRoutingEngine:
    """
    Evaluates incoming prompt signatures to forward tasks to the most effective
    AI provider, based on each provider's free API tier strengths.
    """
    def __init__(self, db_manager: SessionPersistenceManager):
        self.db = db_manager

    def determine_optimal_route(self, prompt: str) -> dict:
        """
        Executes structural token screening across prompt content to dynamically 
        allocate execution pathways based on specialized agent capabilities.
        """
        text_lower = prompt.lower()

        # --- CRITERION 1: HIGH-DENSITY DEVELOPMENT / CODE LOGIC INTENT ---
        if any(term in text_lower for term in ["def ", "class ", "function", "compile", "bug", "sql", "html", "javascript"]):
            log.info("Software development pattern identified. Routing to [Groq].")
            return {
                "agent_name": "Groq Llama 3.3 70B (Stable Attractor)",
                "model_key": "groq",
                "chat_type": "Project",
                "reason": "Groq's hosted open-weight models give fast, low-latency code generation at no cost."
            }

        # --- CRITERION 2: REAL-TIME SEARCH / EXTRACTION INTENT ---
        if any(term in text_lower for term in ["news", "latest", "stock", "weather", "who is", "current", "2026"]):
            log.info("Live data vector identified. Routing to [Tavily].")
            return {
                "agent_name": "Tavily Search Grounding",
                "model_key": "tavily",
                "chat_type": "Stream",
                "reason": "Tavily executes live web search and returns a synthesized, source-grounded answer."
            }

        # --- CRITERION 3: CREATIVE WRITING / HIGH-VOLUME IDEATION ---
        if any(term in text_lower for term in ["write an essay", "brainstorm", "creative story", "poem", "draft", "summarize"]):
            log.info("High-volume generation profile matched. Routing to [Mistral].")
            return {
                "agent_name": "Mistral Small (Limit Cycle Oszillator)",
                "model_key": "mistral",
                "chat_type": "Project",
                "reason": "Mistral's free tier provides flexible linguistic variation for open-ended writing tasks."
            }

        # --- FALLBACK CRITERION: MASSIVE CONTEXT / DEEP TECHNICAL DEPENDENCIES ---
        log.info("Generic multi-turn profile mapped. Allocating to fallback [Gemini].")
        return {
            "agent_name": "Gemini 2.0 Flash (Dissipative Divergence)",
            "model_key": "gemini",
            "chat_type": "Stream",
            "reason": "Gemini's free tier supports a high baseline context window for unsegmented token streams."
        }


log.info("IntentPreRoutingEngine successfully integrated into system core.")