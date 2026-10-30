import numpy as np
import time

def generate_data(n_items, n_features):
    # Large array allocation (hotspot)
    data = np.random.rand(n_items, n_features)
    return data

def compute_scores(data):
    n_items = data.shape[0]

    # Repeated allocation inside loop (hotspot)
    scores = []
    for i in range(n_items):
        # Temporary array allocation (hotspot)
        temp = data[i] * 0.5
        scores.append(np.sum(temp))
    return np.array(scores)

def pairwise_ranking(scores):
    n = len(scores)
    # Nested loop (hotspot)
    ranking_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ranking_matrix[i, j] = scores[i] > scores[j]
    return ranking_matrix

def main():
    start = time.perf_counter()

    data = generate_data(20000, 128)     # Large memory footprint
    scores = compute_scores(data)        # Heavy loop + allocations
    ranking = pairwise_ranking(scores)   # Quadratic nested loop

    end = time.perf_counter()
    print(f"Total runtime: {end - start:.3f} seconds")
    print(f"Ranking matrix shape: {ranking.shape}")

if __name__ == "__main__":
    main()