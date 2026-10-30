"""
cli.py
====================

Command-Line Interface für das Hamilton-Zyklen-Modell.
Ermöglicht das Ausführen der Pipeline mit verschiedenen Generatoren
und Parametern, kompatibel mit Docker, Kubernetes, KServe und Crossplane.
"""

import argparse
from hamilton_model.model import ExtractHamiltonCycles
from hamilton_model.generators import (
    generate_stream_stable,
    generate_stream_drift_training,
    generate_stream_drift_prediction
)


GENERATOR_MAP = {
    "stable": generate_stream_stable,
    "drift_train": generate_stream_drift_training,
    "drift_predict": generate_stream_drift_prediction,
}


def main():
    parser = argparse.ArgumentParser(
        description="Hamilton-Zyklen-Modell CLI"
    )

    parser.add_argument(
        "--generator",
        type=str,
        default="stable",
        choices=GENERATOR_MAP.keys(),
        help="Welcher Datengenerator verwendet werden soll."
    )

    parser.add_argument(
        "--n",
        type=int,
        default=300,
        help="Anzahl der Knoten."
    )

    parser.add_argument(
        "--T",
        type=int,
        default=500,
        help="Anzahl der Zeitschritte."
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=10,
        help="Sampling-Rate für CSV."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Posterior-Schwelle für Kanten."
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plots anzeigen."
    )

    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="CSV-Samples speichern."
    )

    args = parser.parse_args()

    generator = GENERATOR_MAP[args.generator]

    ExtractHamiltonCycles(
        generator=generator,
        n=args.n,
        T=args.T,
        sample_rate=args.sample_rate,
        threshold=args.threshold,
        plot=args.plot,
        save_csv=args.save_csv,
        verbose=True,
        return_results=False
    )


if __name__ == "__main__":
    main()
