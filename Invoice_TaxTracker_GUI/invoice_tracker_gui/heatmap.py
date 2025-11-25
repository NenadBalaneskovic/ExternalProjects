import numpy as np
import matplotlib.pyplot as plt

def render_heatmap(canvas, anomalies):
    grid = np.zeros((5, 5))

    severity_map = {
        "missing": 0.5,
        "formatting": 0.5,
        "mismatch": 1.0,
        "implausible": 1.0,
        "tampering": 1.0,
        "not found": 0.7
    }

    for anomaly in anomalies:
        a = anomaly.lower()
        intensity = next((v for k, v in severity_map.items() if k in a), 0.5)

        if "gesamtpreis" in a:
            grid[1, 0] = intensity
        elif "zahlbetrag" in a:
            grid[1, 1] = intensity
        elif "vat" in a or "ust" in a:
            grid[1, 2] = intensity
        elif "tampering" in a:
            grid[2, 0] = intensity
        elif "missing invoice" in a:
            grid[0, 0] = intensity
        elif "missing vendor" in a:
            grid[0, 1] = intensity
        elif "invalid amount" in a:
            grid[0, 2] = intensity
        elif "address" in a:
            grid[3, 0] = intensity
        else:
            grid[4, 4] = 0.3  # fallback anomaly

    ax = canvas.figure.add_subplot(111)
    ax.clear()
    ax.imshow(grid, cmap="hot", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    canvas.draw()
