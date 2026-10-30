"""
Machine Learning Pipeline Example for MemAlloc Optimizer

This synthetic ML workload intentionally contains:
- large allocations (dataset creation)
- repeated allocations (batch preprocessing)
- temporary arrays (feature transformations)
- nested loops (pairwise distance matrix)
- heavy numerical operations (model training)

It is designed to exercise the full MemAlloc optimization pipeline.
"""

import numpy as np
import time


# ============================================================
# Synthetic Dataset
# ============================================================

def generate_dataset(n_samples=50000, n_features=128):
    """
    Large allocation hotspot:
    Creates a big dense matrix of floats.
    """
    data = np.random.randn(n_samples, n_features)
    labels = (np.sum(data[:, :10], axis=1) > 0).astype(int)
    return data, labels


# ============================================================
# Preprocessing Pipeline
# ============================================================

def preprocess_batch(batch):
    """
    Repeated allocation + temporary arrays:
    Applies normalization and feature scaling.
    """
    # Temporary array
    mean = np.mean(batch, axis=0)
    std = np.std(batch, axis=0) + 1e-6

    # Temporary allocation inside loop
    processed = []
    for row in batch:
        temp = (row - mean) / std
        processed.append(temp)

    return np.array(processed)


def preprocess_dataset(data, batch_size=5000):
    """
    Repeated allocation hotspot:
    Processes dataset in batches.
    """
    n = data.shape[0]
    processed = []

    for i in range(0, n, batch_size):
        batch = data[i:i+batch_size]
        processed.append(preprocess_batch(batch))

    return np.vstack(processed)


# ============================================================
# Simple ML Model (Logistic Regression)
# ============================================================

def train_logistic_regression(X, y, lr=0.01, epochs=10):
    """
    Heavy numerical workload:
    - temporary arrays
    - repeated allocations
    - nested loops (gradient accumulation)
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    for epoch in range(epochs):
        # Temporary allocation
        logits = X @ w
        preds = 1 / (1 + np.exp(-logits))

        # Gradient
        grad = X.T @ (preds - y) / n_samples

        # Update
        w -= lr * grad

    return w


# ============================================================
# Pairwise Distance Matrix (Nested Loop Hotspot)
# ============================================================

def pairwise_distances(X, n_pairs=2000):
    """
    Quadratic nested loop hotspot:
    Computes pairwise distances for a subset.
    """
    subset = X[:n_pairs]
    n = subset.shape[0]

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = subset[i] - subset[j]
            dist[i, j] = np.sqrt(np.sum(diff * diff))

    return dist


# ============================================================
# Main Pipeline
# ============================================================

def main():
    start = time.perf_counter()

    print("Generating dataset...")
    data, labels = generate_dataset()

    print("Preprocessing...")
    X = preprocess_dataset(data)

    print("Training model...")
    w = train_logistic_regression(X, labels)

    print("Computing pairwise distances...")
    dist = pairwise_distances(X)

    end = time.perf_counter()

    print(f"Total runtime: {end - start:.3f} seconds")
    print(f"Model weights shape: {w.shape}")
    print(f"Distance matrix shape: {dist.shape}")


if __name__ == "__main__":
    main()
