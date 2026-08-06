import argparse

from pumpitup.models.baseline import main as run_baseline
from pumpitup.models.tune_all_models import main as run_tuning
from pumpitup.models.stacking_final import main as run_stacking


def main():
    parser = argparse.ArgumentParser(
        description="Pump-It-Up ML Pipeline CLI"
    )

    sub = parser.add_subparsers(dest="command")

    # Baseline CV evaluation
    sub.add_parser("baseline", help="Run baseline CV models")

    # Optuna hyperparameter tuning
    sub.add_parser("tune", help="Run Optuna hyperparameter tuning for all models")

    # Full stacking ensemble training
    sub.add_parser("stack", help="Train full stacking ensemble with tuned parameters")

    # Generate DrivenData submission.csv
    sub.add_parser("submit", help="Generate submission.csv using final stacked model")

    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline()

    elif args.command == "tune":
        run_tuning()

    elif args.command == "stack":
        run_stacking()

    elif args.command == "submit":
        # stacking_final already writes submission.csv
        run_stacking()

    else:
        print("Available commands:")
        print("  baseline  - run baseline CV evaluation")
        print("  tune      - run Optuna hyperparameter tuning")
        print("  stack     - train full stacking ensemble")
        print("  submit    - generate submission.csv")