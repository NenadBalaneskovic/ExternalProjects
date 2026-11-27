# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  api/thread_manager.py
ROLE:    Context gatekeeper synchronizing conversational text states with
         local persistent records and managing the post-generation metric loops.

         Hallucination evaluation uses MULTI-SAMPLE SELF-CONSISTENCY: for each
         prompt, the routed provider is called N_SAMPLES_PER_PROMPT times,
         independently. Every pair among the N responses is compared with the
         exact same SQD / RG-Flow / Graph-Entropy machinery as before
         (core/metrics.py is unchanged); the MEDIAN of each metric across all
         pairs is reported, since a single pair's reading proved too noisy
         (the same "unstable" prompt swung from H=0.06 to H=0.92 across 5
         separate 2-call attempts). Median, not mean, because a handful of
         outlier pairs shouldn't dominate the aggregate the way they would an
         average.
================================================================================
"""

import logging
import statistics
import time
import numpy as np
from core.database import SessionPersistenceManager
from api.web_bridge import BoundedWebSessionBridge, is_error_response
from core.metrics import HeterogeneousDiagnosticEngine

log = logging.getLogger("ThreadContextManager")

# How many independent generations to draw per prompt. More samples -> more
# pairwise comparisons -> a more statistically stable median, but also more
# API calls (free-tier rate limits!) and a longer wait per "Route & Execute".
# 8 samples -> 28 pairs is a reasonable starting balance; raise it if your
# rate limits allow and you want tighter statistics.
N_SAMPLES_PER_PROMPT = 8

# Small delay between successive generation calls to the same provider, to
# avoid bursting a free-tier per-minute rate limit when firing N requests in
# quick succession. _post_with_retry in web_bridge.py already retries on
# 429/503, but spacing the calls out reduces how often that's even needed.
INTER_SAMPLE_DELAY_SECONDS = 0.6

_NO_COMPARISON_AVAILABLE = {
    "h_score": 0.0,
    "is_definitive_hallucination": False,
    "mean_drift_x": 0.0,
    "trace_covariance_y": 0.0,
    "raw_metrics": {"sqd": 0.0, "rg_instability": 0.0, "graph_entropy_drift": 0.0},
}


class ThreadContextOrchestrator:
    """
    Coordinates multi-turn conversational text states and maps subsequent prompts
    to the appropriate provider, then runs the multi-sample self-consistency
    hallucination evaluation across N independent generations for that prompt.
    """
    def __init__(self, db: SessionPersistenceManager, bridge: BoundedWebSessionBridge, engine: HeterogeneousDiagnosticEngine):
        self.db = db
        self.bridge = bridge
        self.engine = engine

    def process_interaction_step(self, session_id: int, prompt: str, agent_meta: dict, progress_callback=None) -> dict:
        """
        Draws N_SAMPLES_PER_PROMPT independent generations for `prompt` from
        the routed provider, evaluates every pair with the diagnostic engine,
        and aggregates each metric via its MEDIAN across all pairs.

        `progress_callback`, if given, is called with a short status string
        after each generation (e.g. "Sample 3/8 erzeugt...") so the GUI can
        show progress during what is now a multi-call, multi-second step.
        """
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT step FROM metrics_log 
                WHERE session_id = ? ORDER BY step DESC LIMIT 1
            """, (session_id,))
            last_log = cursor.fetchone()
        next_step = (last_log["step"] + 1) if last_log else 1

        log.info(f"Step {next_step}: drawing {N_SAMPLES_PER_PROMPT} independent self-consistency samples.")

        # 1. Draw N independent generations.
        samples = []
        for i in range(N_SAMPLES_PER_PROMPT):
            text = self.bridge.forward_prompt_to_endpoint(agent_meta, prompt, thread_url="self-consistency")
            if is_error_response(text):
                log.warning(f"Sample {i + 1}/{N_SAMPLES_PER_PROMPT} failed: {text}")
            else:
                samples.append(text)
            if progress_callback:
                progress_callback(f"Sample {i + 1}/{N_SAMPLES_PER_PROMPT} erzeugt...")
            if i < N_SAMPLES_PER_PROMPT - 1:
                time.sleep(INTER_SAMPLE_DELAY_SECONDS)

        if len(samples) < 2:
            log.warning("Fewer than 2 successful samples; cannot evaluate self-consistency for this step.")
            diagnostic_payload = dict(_NO_COMPARISON_AVAILABLE)
            response_text = samples[0] if samples else "[ERROR] Alle Self-Consistency-Samples sind fehlgeschlagen."
        else:
            # 2. Embed every successful sample (one batched embedding call per sample).
            if progress_callback:
                progress_callback("Erzeuge Embeddings für alle Samples...")
            embeddings_list = [self.bridge.get_embedding_ensemble(s) for s in samples]

            # 3. Evaluate every pair with the UNCHANGED diagnostic engine.
            if progress_callback:
                progress_callback("Werte alle Sample-Paare aus...")
            pair_results = []
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    pair_results.append(
                        self.engine.evaluate_response_coherence(
                            embeddings_list[i], embeddings_list[j], samples[i], samples[j]
                        )
                    )

            diagnostic_payload = self._aggregate_median(pair_results)
            response_text = self._select_medoid(samples, embeddings_list)
            log.info(
                f"Aggregated {len(pair_results)} pairs from {len(samples)} samples. "
                f"Median H-Score: {diagnostic_payload['h_score']:.4f}"
            )

        # 4. Persist the representative (medoid) response and the aggregated diagnostic.
        self.db.log_trajectory_step(session_id, next_step, prompt, response_text, diagnostic_payload)

        return {
            "step": next_step,
            "response_text": response_text,
            "diagnostic": diagnostic_payload,
            "sample_count": len(samples),
        }

    @staticmethod
    def _aggregate_median(pair_results: list) -> dict:
        """Collapses a list of per-pair diagnostic dicts into one dict of medians."""
        h_scores = [r["h_score"] for r in pair_results]
        drift_x = [r["mean_drift_x"] for r in pair_results]
        cov_y = [r["trace_covariance_y"] for r in pair_results]
        sqd = [r["raw_metrics"]["sqd"] for r in pair_results]
        rg = [r["raw_metrics"]["rg_instability"] for r in pair_results]
        topo = [r["raw_metrics"]["graph_entropy_drift"] for r in pair_results]

        # Majority vote across pairs, rather than re-deriving metrics.py's
        # three-condition threshold here (which would duplicate and risk
        # drifting out of sync with the thresholds defined there).
        definitive_votes = sum(1 for r in pair_results if r["is_definitive_hallucination"])
        is_definitive = definitive_votes > (len(pair_results) / 2)

        return {
            "h_score": statistics.median(h_scores),
            "is_definitive_hallucination": is_definitive,
            "mean_drift_x": statistics.median(drift_x),
            "trace_covariance_y": statistics.median(cov_y),
            "raw_metrics": {
                "sqd": statistics.median(sqd),
                "rg_instability": statistics.median(rg),
                "graph_entropy_drift": statistics.median(topo),
            },
        }

    @staticmethod
    def _select_medoid(samples: list, embeddings_list: list) -> str:
        """
        Picks the sample whose per-response centroid embedding is closest
        (cosine distance) to the mean of all samples' centroids - i.e. the
        most "representative" of the N independent generations, rather than
        arbitrarily always showing the first one.
        """
        centroids = np.array([np.mean(e, axis=0) for e in embeddings_list])
        global_mean = np.mean(centroids, axis=0)
        global_norm = np.linalg.norm(global_mean) + 1e-8

        best_idx, best_dist = 0, float("inf")
        for idx, c in enumerate(centroids):
            cos_dist = 1.0 - (np.dot(c, global_mean) / ((np.linalg.norm(c) + 1e-8) * global_norm))
            if cos_dist < best_dist:
                best_idx, best_dist = idx, cos_dist

        return samples[best_idx]


log.info("ThreadContextOrchestrator successfully coupled to the network controller layer.")