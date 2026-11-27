# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  core/metrics.py
ROLE:    Advanced mathematical evaluation framework computing Semantic Quantum 
         Drift (SQD), RG Coarse-Graining, and stationary PageRank Graph Entropy.
================================================================================
"""

import re
import logging
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA

log = logging.getLogger("MetricsEngine")

# A variance floor used when regularizing the KL divergence. The previous
# value (1e-6) was several orders of magnitude smaller than the typical
# per-dimension variance of a normalized embedding ensemble (~1e-4 to 1e-2).
# Whenever an ensemble happened to have near-zero intrinsic variance (e.g.
# repeated/duplicate samples of the same text), the (mu_t - mu_0)^2 / s_t
# term divided by a near-zero denominator and blew up to values in the
# hundreds of thousands, which in turn saturated every downstream score.
# This floor is sized relative to the embedding scale instead of being an
# arbitrary absolute constant close to machine precision.
DEFAULT_VARIANCE_FLOOR = 1e-3

# Hard ceiling on a single KL computation. Even with a sane variance floor,
# pathological inputs (e.g. a genuinely empty or degenerate ensemble) could
# still in principle produce a very large divergence. Capping it keeps every
# downstream quantity (rg_instability, combined_amplitude, tanh(...)) well
# defined instead of propagating an unbounded number through the pipeline.
MAX_KL_VALUE = 50.0


class QuantumRGDriftAnalyzer:
    """
    Core mathematical module executing State-Space and Topographical measurements 
    to track semantic trajectory deformations over discrete execution steps.
    """
    def __init__(self, embedding_dim: int = 1536):
        self.d = embedding_dim

    def calculate_regularized_kl(self, mu_0: np.ndarray, sigma_0: np.ndarray,
                                 mu_t: np.ndarray, sigma_t: np.ndarray,
                                 epsilon: float = DEFAULT_VARIANCE_FLOOR) -> float:
        """
        Computes the analytical Kullback-Leibler (KL) Divergence D_KL(P_0 || Q_t) between 
        two multivariate Gaussian response distributions using a regularized diagonal structure.
        
        Formula:
        D_KL = 0.5 * sum( (sigma_0^2 / sigma_t^2) + ((mu_t - mu_0)^2 / sigma_t^2) - 1 + ln(sigma_t^2 / sigma_0^2) )

        `epsilon` acts as a variance FLOOR (via np.maximum), not a small additive
        nudge: it guarantees sigma_0/sigma_t never drop far enough below the
        embedding's natural variance scale to make the division terms explode.
        Using np.maximum instead of a plain "+ epsilon" matters because an
        additive epsilon of this size would otherwise materially bias the
        trace/logdet terms even for perfectly healthy, non-degenerate ensembles.
        """
        s_0 = np.maximum(sigma_0, epsilon)
        s_t = np.maximum(sigma_t, epsilon)

        # High-performance vectorized calculation across the embedding subspace
        trace_term = np.sum(s_0 / s_t)
        distance_term = np.dot((mu_t - mu_0) ** 2, 1.0 / s_t)
        dim_term = -len(mu_0)
        logdet_term = np.sum(np.log(s_t / s_0))

        kl_value = 0.5 * (trace_term + distance_term + dim_term + logdet_term)
        kl_value = max(0.0, float(kl_value))

        # Normalize to a PER-DIMENSION average instead of a raw sum. The
        # number of dimensions here (micro_k / macro_k in compute_rg_flow) is
        # itself a function of how many sentences the response has, which has
        # nothing to do with how consistent the content is. Without this
        # normalization, a longer (but equally correct) answer automatically
        # projects onto more PCA components and therefore accumulates a
        # larger summed KL, independent of actual semantic drift.
        kl_value = kl_value / max(1, len(mu_0))

        # Clamp instead of silently trusting an unbounded result; log so a
        # persistently saturated value is visible rather than masquerading
        # as a normal high score.
        if kl_value > MAX_KL_VALUE:
            log.warning(
                f"Regularized KL divergence {kl_value:.2f} exceeded the {MAX_KL_VALUE} "
                f"safety ceiling and was clamped. This usually means the input ensembles "
                f"are degenerate (e.g. near-zero variance) rather than reflecting a "
                f"genuinely extreme semantic drift."
            )
            kl_value = MAX_KL_VALUE

        return kl_value

    def compute_rg_flow(self, ensemble_0: np.ndarray, ensemble_t: np.ndarray) -> tuple:
        """
        Applies a multi-scale Coarse-Graining operator via dynamic PCA to compare
        microscopic (Lambda_0) vs macroscopic (Lambda_1) information dissipation.
        Returns the RG instability coefficient (gamma_RG) and the raw microscopic drift score.
        """
        # Determine rigorous matrix boundaries for BOTH projections up front.
        # Embeddings now come from a real embedding model (hundreds to
        # thousands of dimensions, d), while each ensemble only has as many
        # samples as the text has sentences (n, typically 5-15). Estimating a
        # per-dimension variance directly in the raw d-dimensional space with
        # n << d samples is a classic small-sample/high-dimension problem:
        # the estimate is essentially noise, and it made the KL divergence
        # explode on any two independently-written (but equally valid) texts.
        #
        # Fix: bound BOTH the "microscopic" and "macroscopic" scales by the
        # sample count via PCA, exactly like the macroscopic scale already
        # was. Micro gets the full statistically resolvable rank (as fine-
        # grained as the data actually supports); macro gets a small,
        # deliberately coarse-grained rank. The ratio between them is then a
        # genuine measure of RG coarse-graining instability, independent of
        # the embedding model's raw dimensionality.
        combined_matrix = np.vstack([ensemble_0, ensemble_t])
        n_samples, d = combined_matrix.shape
        max_resolvable_rank = max(1, min(n_samples - 1, d))

        if max_resolvable_rank < 2:
            # Not enough independent samples (e.g. a one-sentence response)
            # to resolve any meaningful covariance structure at all.
            return 0.0, 0.0

        micro_k = max_resolvable_rank
        macro_k = max(1, min(4, max_resolvable_rank))  # deliberately coarse

        # 1. Microscopic Scale: project to the full rank the sample size can
        #    actually support (this REPLACES computing directly in the raw
        #    embedding space, which was the source of the blowup).
        pca_micro = PCA(n_components=micro_k)
        pca_micro.fit(combined_matrix)
        cg_0_micro = pca_micro.transform(ensemble_0)
        cg_t_micro = pca_micro.transform(ensemble_t)

        mu_0_micro = np.mean(cg_0_micro, axis=0)
        sigma_0_micro = np.var(cg_0_micro, axis=0)
        mu_t_micro = np.mean(cg_t_micro, axis=0)
        sigma_t_micro = np.var(cg_t_micro, axis=0)

        kl_micro = self.calculate_regularized_kl(mu_0_micro, sigma_0_micro, mu_t_micro, sigma_t_micro)

        if macro_k >= micro_k:
            # Degenerate case (very few samples): no room left for a coarser
            # projection than the micro one, so there is nothing to compare.
            return 0.0, kl_micro

        # 2. Macroscopic Scale: a deliberately smaller principal subspace of
        #    the SAME combined_matrix, representing the coarse-grained view.
        pca_macro = PCA(n_components=macro_k)
        pca_macro.fit(combined_matrix)

        cg_0 = pca_macro.transform(ensemble_0)
        cg_t = pca_macro.transform(ensemble_t)

        # Macroscopic Parameter Extraction
        mu_0_macro = np.mean(cg_0, axis=0)
        sigma_0_macro = np.var(cg_0, axis=0)
        mu_t_macro = np.mean(cg_t, axis=0)
        sigma_t_macro = np.var(cg_t, axis=0)

        # Compute macroscopic divergence on scale Lambda_1
        kl_macro = self.calculate_regularized_kl(mu_0_macro, sigma_0_macro, mu_t_macro, sigma_t_macro)

        # Calculate the RG Instability Coefficient gamma_RG.
        # kl_micro is now floor-protected (never near-zero in a degenerate
        # way without also capping kl_macro), so this ratio no longer swings
        # to extreme values purely from denominator collapse.
        rg_instability = kl_macro / (kl_micro + 1e-3)
        return float(rg_instability), float(kl_micro)

    @staticmethod
    def _strip_non_prose_markup(text: str) -> str:
        """
        Removes fenced/inline code, LaTeX math and markdown structural markup
        before natural-language tokenization. These are legitimate parts of a
        good answer, but they are not prose: variable names, operators and
        braces inflate the co-occurrence graph with tokens that vary wildly
        between two independently-written (yet equally correct) answers,
        which previously made graph_entropy_drift noisy and oversensitive.
        """
        cleaned = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)   # fenced code blocks
        cleaned = re.sub(r'`[^`]*`', ' ', cleaned)                    # inline code
        cleaned = re.sub(r'\\\[.*?\\\]', ' ', cleaned, flags=re.DOTALL)  # \[ ... \] LaTeX
        cleaned = re.sub(r'\$[^$]*\$', ' ', cleaned)                  # $...$ LaTeX
        cleaned = re.sub(r'^\s{0,3}#{1,6}\s*', '', cleaned, flags=re.MULTILINE)  # markdown headers
        cleaned = re.sub(r'^\s*[-*+]\s+', '', cleaned, flags=re.MULTILINE)       # bullet markers
        cleaned = re.sub(r'^-{3,}\s*$', ' ', cleaned, flags=re.MULTILINE)        # horizontal rules
        return cleaned

    def compute_topological_graph_entropy(self, text: str) -> float:
        """
        Constructs a localized word co-occurrence network using a sliding token window.
        Extracts structural Shannon graph entropy weighted by the network's stationary PageRank density.
        """
        # Tokenize inputs, stripping punctuation and filtering structure words.
        # Code blocks, inline code, markdown markup and LaTeX are stripped
        # first (see _strip_non_prose_markup): identifiers, operators and
        # braces are not natural-language tokens, and two independently
        # written (but equally correct) code answers will differ hugely in
        # these tokens even when the actual prose explanation is consistent.
        prose_only = self._strip_non_prose_markup(text)
        tokens = re.findall(r'\b\w+\b', prose_only.lower())
        if len(tokens) < 3:
            return 0.0

        G = nx.DiGraph()

        # Populate directed network edges within a sliding window framework (Width = 3)
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            for src, dst in [(w1, w2), (w2, w3), (w1, w3)]:
                if G.has_edge(src, dst):
                    G[src][dst]['weight'] += 1
                else:
                    G.add_edge(src, dst, weight=1)

        if G.number_of_edges() == 0:
            return 0.0

        # Extract stationary probability density vector rho via Power Iteration (PageRank)
        try:
            pagerank = nx.pagerank(G, weight='weight', max_iter=100, tol=1e-5)
        except Exception:
            # Numerical fallback to uniform node distribution if network iteration fails to converge
            pagerank = {node: 1.0 / G.number_of_nodes() for node in G.nodes()}

        graph_entropy = 0.0
        for node in G.nodes():
            rho_i = pagerank[node]
            successors = list(G.successors(node))
            total_out_weight = sum([G[node][succ]['weight'] for succ in successors])

            if total_out_weight == 0:
                continue

            node_entropy = 0.0
            for succ in successors:
                p_ij = G[node][succ]['weight'] / total_out_weight
                if p_ij > 0:
                    node_entropy -= p_ij * np.log(p_ij)

            # Weigh localized textual entropy against global central node authority (PageRank)
            graph_entropy += rho_i * node_entropy

        return float(graph_entropy)


class HeterogeneousDiagnosticEngine:
    """
    Orchestrates multiple independent geometric and topographical data streams 
    to output an un-biased, scalarised H-Score using bounded hyperbolic curves.
    """
    def __init__(self, analyzer: QuantumRGDriftAnalyzer):
        self.analyzer = analyzer

    def evaluate_response_coherence(self, base_ensemble: np.ndarray, current_ensemble: np.ndarray,
                                    base_text: str, current_text: str) -> dict:
        """
        Processes parallel state spaces to determine continuous hallucination markers
        and flag critical state transitions for automated failover routines.
        """
        # 1. Extract Micro vs Macro information loss metrics via the RG Operator
        rg_instability, sqd_value = self.analyzer.compute_rg_flow(base_ensemble, current_ensemble)

        # 2. Extract topological graph text data profiles
        s_g0 = self.analyzer.compute_topological_graph_entropy(base_text)
        s_gt = self.analyzer.compute_topological_graph_entropy(current_text)
        # Relative drift instead of a raw absolute difference: an absolute
        # gap grows naturally with vocabulary size/text length regardless of
        # real inconsistency, whereas this ratio stays bounded in [0, 1] and
        # is comparable across answers of very different length.
        graph_entropy_drift = abs(s_gt - s_g0) / (max(s_g0, s_gt) + 1e-6)

        # 3. Calculate translational spatial shift parameters
        mu_0 = np.mean(base_ensemble, axis=0)
        mu_t = np.mean(current_ensemble, axis=0)

        # High-precision normalized cosine distance calculation
        norm_product = np.linalg.norm(mu_0) * np.linalg.norm(mu_t)
        mean_drift = float(1.0 - (np.dot(mu_0, mu_t) / (norm_product + 1e-8)))

        # Extract instantane variance blur (Trace of the Ensemble Covariance Matrix)
        trace_covariance = float(np.sum(np.var(current_ensemble, axis=0)))

        # --- HETEROGENEOUS CALIBRATED THRESHOLD VERIFICATION NETWORK ---
        # These three thresholds and the SQD_AMPLITUDE_SCALE below were
        # originally tuned for sqd_value as a raw SUMMED KL divergence over
        # ~1536 synthetic dimensions. Now that calculate_regularized_kl
        # returns a PER-DIMENSION AVERAGE (see that method), sqd_value lives
        # on a much smaller natural scale, so these constants were lowered
        # accordingly. There is no universally "correct" value for them -
        # they encode where YOU want the SECURE/BLUR/CRITICAL line to sit for
        # your own experiment. Treat them as tunable knobs: run a batch of
        # prompts you consider genuinely stable/well-grounded vs. ones you
        # consider genuinely ambiguous/inconsistent, look at the resulting
        # raw_metrics, and adjust these constants to match where you want the
        # boundary to fall for your own notion of "hallucination".
        SQD_DEFINITIVE_THRESHOLD = 0.5
        RG_DEFINITIVE_THRESHOLD = 0.4
        GRAPH_ENTROPY_DEFINITIVE_THRESHOLD = 0.15
        SQD_AMPLITUDE_SCALE = 0.5  # divisor applied to sqd_value before combining

        # Verification matrix: Flag True ONLY if all three tracking signals breach boundaries simultaneously
        is_definitive = (
            (sqd_value > SQD_DEFINITIVE_THRESHOLD)
            and (rg_instability > RG_DEFINITIVE_THRESHOLD)
            and (graph_entropy_drift > GRAPH_ENTROPY_DEFINITIVE_THRESHOLD)
        )

        # Compile continuous telemetry score bounded strictly to [0.0, 1.0] via tanh().
        # sqd_value and rg_instability are now both floor/ceiling protected
        # upstream, so this amplitude reflects genuine drift rather than a
        # division-by-near-zero or text-length artifact.
        combined_amplitude = (sqd_value / SQD_AMPLITUDE_SCALE) * rg_instability * (1.0 + graph_entropy_drift)
        hallucination_grade = float(np.tanh(combined_amplitude))

        return {
            "h_score": hallucination_grade,
            "is_definitive_hallucination": is_definitive,
            "mean_drift_x": mean_drift,
            "trace_covariance_y": trace_covariance,
            "raw_metrics": {
                "sqd": sqd_value,
                "rg_instability": rg_instability,
                "graph_entropy_drift": graph_entropy_drift
            }
        }

log.info("Production core/metrics.py primary mathematical modules locked into system core.")