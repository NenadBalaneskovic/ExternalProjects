# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  api/web_bridge.py
ROLE:    Network transport layer calling the official free-tier APIs of
         Gemini, Groq, Mistral and Tavily using API keys (no browser cookies,
         no scraping of consumer web UIs). Also supplies real text embeddings
         (via Gemini's embedding endpoint) for the local metrics engine.
================================================================================
"""

import os
import logging
import re
import time
import numpy as np
import requests
from dotenv import load_dotenv

log = logging.getLogger("WebBridgeEngine")

# NOTE on model names: providers rename/rotate their free-tier model
# identifiers from time to time. If a call below fails with a 404 / "model
# not found" style error, check the provider's current docs/console and
# update the constant here - the request/response shape will stay the same.
# NOTE: gemini-2.0-flash was shut down by Google on 2026-06-01. Current
# recommended replacement per ai.google.dev/gemini-api/docs/deprecations:
GEMINI_MODEL = "gemini-3.6-flash"
# NOTE: Google renamed its embedding model; the old "text-embedding-004"
# returns 404 Not Found. Current model per ai.google.dev/api/embeddings:
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
# NOTE: llama-3.3-70b-versatile was decommissioned by Groq on 2026-08-16.
# Current recommended replacement per console.groq.com/docs/deprecations:
GROQ_MODEL = "openai/gpt-oss-120b"
MISTRAL_MODEL = "mistral-small-latest"

# forward_prompt_to_endpoint() reports provider failures as a returned string
# (not an exception) so the GUI can display them inline. This prefix list lets
# callers (e.g. the self-consistency pairing in thread_manager.py) detect that
# case without re-parsing error text.
ERROR_PREFIXES = ("[TRANSPORT ERROR]", "[CONFIG ERROR]")


def is_error_response(text: str) -> bool:
    return any(text.startswith(p) for p in ERROR_PREFIXES)


REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRY_BACKOFF_SECONDS = 3


def _post_with_retry(url: str, **kwargs) -> requests.Response:
    """
    Wraps requests.post with automatic retries on:
      - a read/connect timeout, and
      - transient server-side errors (429 rate limit, 5xx overload) which
        are common on free-tier endpoints under load and usually clear up
        within a few seconds, with no fault in the request itself.
    Non-retryable errors (e.g. 404 bad model name, 401 bad key) are raised
    immediately via raise_for_status() by the caller, since retrying those
    would just waste time on something that isn't going to change.
    """
    attempt = 0
    while True:
        try:
            resp = requests.post(url, **kwargs)
            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                attempt += 1
                log.warning(
                    f"Request to {url} returned {resp.status_code} (likely transient "
                    f"provider overload); retrying in {RETRY_BACKOFF_SECONDS}s "
                    f"({attempt}/{MAX_RETRIES})..."
                )
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            return resp
        except requests.exceptions.Timeout:
            if attempt >= MAX_RETRIES:
                raise
            attempt += 1
            log.warning(f"Request to {url} timed out; retrying ({attempt}/{MAX_RETRIES})...")


class BoundedWebSessionBridge:
    """
    Handles network requests to AI provider APIs using official, user-supplied
    API keys. No browser automation, no session-cookie harvesting.
    """
    def __init__(self):
        # Resolve path to the global environment configuration file
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.env_path = os.path.join(self.base_dir, '.env')
        self.synchronize_tokens()

    def synchronize_tokens(self):
        """Reloads active API keys from the filesystem environment state wrapper."""
        if os.path.exists(self.env_path):
            load_dotenv(self.env_path, override=True)

        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.mistral_key = os.getenv("MISTRAL_API_KEY")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        log.info("Active API credentials synchronized from dynamic environment storage.")

    def forward_prompt_to_endpoint(self, agent_meta: dict, prompt: str, thread_url: str = None) -> str:
        """
        Dispatches the prompt to whichever provider the router selected.
        `thread_url` is accepted for interface compatibility with the
        orchestrator but these stateless free-tier APIs do not maintain a
        provider-side thread, so it is not transmitted anywhere.
        """
        self.synchronize_tokens()
        key = agent_meta["model_key"]

        try:
            if key == "groq":
                return self._execute_groq_transaction(prompt)
            elif key == "mistral":
                return self._execute_mistral_transaction(prompt)
            elif key == "gemini":
                return self._execute_gemini_transaction(prompt)
            elif key == "tavily":
                return self._execute_tavily_transaction(prompt)
            else:
                raise ValueError(f"Unknown routing token signature encountered: {key}")
        except requests.exceptions.RequestException as e:
            log.error(f"Network transport level exception caught during communication: {str(e)}")
            return f"[TRANSPORT ERROR] Connection to [{key}] failed: {str(e)}"
        except ValueError as e:
            log.error(f"Provider configuration error: {str(e)}")
            return f"[CONFIG ERROR] {str(e)}"

    # ------------------------------------------------------------ providers
    def _execute_groq_transaction(self, prompt: str) -> str:
        """Groq's OpenAI-compatible chat completions endpoint (free tier)."""
        if not self.groq_key:
            raise ValueError("GROQ_API_KEY missing. Add it via the API-Key Sync Center.")

        resp = _post_with_retry(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}]},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        log.info(f"Groq ({GROQ_MODEL}) responded successfully.")
        return text

    def _execute_mistral_transaction(self, prompt: str) -> str:
        """Mistral's chat completions endpoint (free tier)."""
        if not self.mistral_key:
            raise ValueError("MISTRAL_API_KEY missing. Add it via the API-Key Sync Center.")

        resp = _post_with_retry(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.mistral_key}", "Content-Type": "application/json"},
            json={"model": MISTRAL_MODEL, "messages": [{"role": "user", "content": prompt}]},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        log.info(f"Mistral ({MISTRAL_MODEL}) responded successfully.")
        return text

    def _execute_gemini_transaction(self, prompt: str) -> str:
        """Gemini's generateContent endpoint (free tier)."""
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY missing. Add it via the API-Key Sync Center.")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={self.gemini_key}"
        )
        resp = _post_with_retry(
            url,
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()

        # A 200 OK response does not guarantee a usable candidate: Gemini can
        # block a prompt on input safety grounds (promptFeedback.blockReason,
        # no "candidates" at all) or truncate/block an individual candidate
        # (finishReason "SAFETY"/"RECITATION"/"OTHER", with no "parts"). Both
        # cases previously caused an unhandled KeyError/IndexError here,
        # which forward_prompt_to_endpoint's except clauses do NOT catch
        # (they only catch RequestException/ValueError) - so a safety block
        # used to surface as a raw crash instead of a clean error message.
        block_reason = data.get("promptFeedback", {}).get("blockReason")
        if block_reason:
            raise ValueError(f"Gemini blocked this prompt before generation (reason: {block_reason}).")

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates (likely blocked; no reason given in response).")

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        parts = candidate.get("content", {}).get("parts", [])
        if not parts:
            raise ValueError(f"Gemini candidate had no content (finishReason: {finish_reason}).")

        text = parts[0].get("text", "")
        if not text:
            raise ValueError(f"Gemini candidate had an empty text part (finishReason: {finish_reason}).")

        log.info(f"Gemini ({GEMINI_MODEL}) responded successfully.")
        return text

    def _execute_tavily_transaction(self, prompt: str) -> str:
        """Tavily's search endpoint, requesting a synthesized answer (free tier)."""
        if not self.tavily_key:
            raise ValueError("TAVILY_API_KEY missing. Add it via the API-Key Sync Center.")

        # Current Tavily auth is a Bearer header (docs.tavily.com), not an
        # "api_key" field inside the JSON body as in older examples.
        resp = _post_with_retry(
            "https://api.tavily.com/search",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.tavily_key}",
            },
            json={
                "query": prompt,
                "search_depth": "basic",
                "include_answer": True,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer")
        if answer:
            log.info("Tavily returned a synthesized, source-grounded answer.")
            return answer

        # Fall back to concatenating top result snippets if no synthesized answer came back.
        results = data.get("results", [])
        if results:
            log.info("Tavily returned raw results (no synthesized answer); concatenating snippets.")
            return "\n\n".join(r.get("content", "") for r in results[:3])

        return "[TAVILY] No results found for this query."

    # ------------------------------------------------------------ embeddings
    def get_embedding_ensemble(self, text: str) -> np.ndarray:
        """
        Returns a real embedding ensemble for `text`: one Gemini embedding
        vector per sentence (up to 8), via a single batched API call. This
        reflects genuine internal variance of the text instead of the old
        mock approach (identical vectors with fabricated randomness).

        An empty/whitespace-only text returns a single zero vector; this is
        handled gracefully by core/metrics.py's variance floor and distance
        regularization, so callers don't need special-case logic.
        """
        self.synchronize_tokens()
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY missing; required for embeddings regardless of which provider generated the text.")

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        if not sentences:
            # No content yet (e.g. very first turn's "base_text"). Embed a
            # neutral placeholder instead of returning a hardcoded zero
            # vector: the embedding model's output dimensionality can change
            # between models/versions, and a mismatched hardcoded shape here
            # would make (mu_t - mu_0) in core/metrics.py fail with a numpy
            # broadcasting error the moment a real embedding is involved.
            sentences = ["(no prior response yet)"]
        else:
            sentences = sentences[:8]  # cap request size / API usage

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_EMBEDDING_MODEL}:batchEmbedContents?key={self.gemini_key}"
        )
        body = {
            "requests": [
                {
                    "model": f"models/{GEMINI_EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": s}]},
                }
                for s in sentences
            ]
        }
        resp = _post_with_retry(url, headers={"Content-Type": "application/json"}, json=body, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        vectors = [e["values"] for e in data["embeddings"]]
        return np.array(vectors, dtype=float)


log.info("BoundedWebSessionBridge initialized and locked into communication profile.")