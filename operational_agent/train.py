"""Train the ED/Operational agent and write metrics the team can compare."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from operational_agent.data import (
    DATASET_URL,
    component_leakage_report,
    load_visits,
    long_wait_label,
    prepare_features,
)
from operational_agent.features import (
    CV_FOLDS,
    DEFAULT_LONG_WAIT_MINUTES,
    MODEL_FEATURES,
    RANDOM_STATE,
    SOURCE_FEATURES,
    TEAM_FEATURE_SPLIT,
    TEST_SIZE,
)
from operational_agent.model import (
    OperationalAgent,
    classification_metrics,
    default_estimators,
    select_model,
)


def train_operational_agent(
    visits: pd.DataFrame,
    output_dir: str | Path = "outputs/operational",
    threshold_minutes: float = DEFAULT_LONG_WAIT_MINUTES,
    threshold_quantile: float | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv_folds: int = CV_FOLDS,
    model_name: str = "auto",
    estimators: dict | None = None,
) -> dict:
    """Fit the agent, score it on held-out visits, and save the coordinator artifact.

    The row split uses only ``random_state`` and the current row order, so the
    patient agent, triage agent, and baseline can score the same visits. A
    quantile cutoff is computed on the training rows only.
    """
    if "total_time" not in visits.columns:
        raise ValueError("The dataset needs a total_time column to build the long-wait label.")
    if threshold_quantile is not None and not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be between 0 and 1.")

    total_time = pd.to_numeric(visits["total_time"], errors="coerce")
    train_frame, test_frame = train_test_split(
        visits,
        test_size=test_size,
        random_state=random_state,
    )
    if threshold_quantile is None:
        cutoff = float(threshold_minutes)
        cutoff_source = "fixed_minutes"
    else:
        cutoff = float(pd.to_numeric(train_frame["total_time"], errors="coerce").quantile(threshold_quantile))
        cutoff_source = "training_quantile"

    y_train = long_wait_label(train_frame["total_time"], cutoff)
    y_test = long_wait_label(test_frame["total_time"], cutoff)
    x_train = prepare_features(train_frame)
    x_test = prepare_features(test_frame)

    catalog = estimators if estimators is not None else default_estimators(random_state)
    if model_name != "auto":
        if model_name not in catalog:
            known = ", ".join(sorted(catalog))
            raise ValueError(f"Unknown model '{model_name}'. Choose one of: {known}, or auto.")
        catalog = {model_name: catalog[model_name]}

    selected_name, pipeline, selection = select_model(
        x_train,
        y_train,
        catalog,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    agent = OperationalAgent(pipeline, selected_name, cutoff)
    probabilities = agent.predict_proba(test_frame)
    test_metrics = classification_metrics(y_test, probabilities)
    importance = _column_importance(pipeline, x_test, y_test, random_state)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "operational_agent.joblib"
    agent.save(model_path)
    predictions = _prediction_frame(test_frame, y_test, probabilities)
    predictions_path = destination / "test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    split_path = destination / "split_visit_ids.json"
    split_payload = _split_payload(train_frame, test_frame, cutoff, cutoff_source, random_state, test_size)
    split_path.write_text(json.dumps(split_payload, indent=2))

    report = {
        "agent": "ed_operational",
        "features": MODEL_FEATURES,
        "source_columns": SOURCE_FEATURES,
        "team_feature_split": TEAM_FEATURE_SPLIT,
        "target": {
            "name": "long_wait",
            "rule": f"total_time >= {cutoff:.4f} minutes",
            "threshold_minutes": cutoff,
            "threshold_source": cutoff_source,
            "observed_total_time_min": float(total_time.min()),
            "observed_total_time_max": float(total_time.max()),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "split": {
            "random_state": random_state,
            "test_size": test_size,
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
        },
        "model_selection_roc_auc": selection,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "permutation_importance_roc_auc": importance,
        "wait_component_check": component_leakage_report(visits),
        "artifacts": {
            "model": str(model_path),
            "test_predictions": str(predictions_path),
            "split_visit_ids": str(split_path),
        },
    }
    metrics_path = destination / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2))
    report["artifacts"]["metrics"] = str(metrics_path)
    _save_plots(destination, y_test, probabilities, importance)
    return report


def _prediction_frame(frame: pd.DataFrame, labels: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    visit_ids = frame["visit_id"] if "visit_id" in frame.columns else frame.index.astype(str)
    return pd.DataFrame(
        {
            "visit_id": visit_ids.to_numpy(),
            "total_time": pd.to_numeric(frame["total_time"], errors="coerce").to_numpy(),
            "long_wait": labels.to_numpy(),
            "operational_probability": probabilities,
            "operational_prediction": (probabilities >= 0.5).astype(int),
        }
    )


def _split_payload(train_frame, test_frame, cutoff, cutoff_source, random_state, test_size) -> dict:
    def ids(frame: pd.DataFrame) -> list[str]:
        if "visit_id" in frame.columns:
            return frame["visit_id"].astype(str).tolist()
        return [str(index) for index in frame.index.tolist()]

    return {
        "agent": "ed_operational",
        "random_state": random_state,
        "test_size": test_size,
        "threshold_minutes": cutoff,
        "threshold_source": cutoff_source,
        "train_visit_ids": ids(train_frame),
        "test_visit_ids": ids(test_frame),
    }


def _column_importance(pipeline, features, labels, random_state: int) -> dict[str, float]:
    result = permutation_importance(
        pipeline,
        features,
        labels,
        scoring="roc_auc",
        n_repeats=10,
        random_state=random_state,
        n_jobs=1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    return {
        features.columns[index]: {
            "mean_roc_auc_drop": float(result.importances_mean[index]),
            "std": float(result.importances_std[index]),
        }
        for index in order
    }


def _save_plots(destination: Path, labels, probabilities, importance: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay
    except ImportError:
        return
    roc_path = destination / "roc_curve.png"
    display = RocCurveDisplay.from_predictions(labels, probabilities, name="ED/Operational agent")
    display.ax_.set_title("ED/Operational agent ROC")
    display.figure_.tight_layout()
    display.figure_.savefig(roc_path, dpi=120)
    plt.close(display.figure_)

    names = list(importance)
    drops = [importance[name]["mean_roc_auc_drop"] for name in names]
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.barh(names[::-1], drops[::-1], color="#2c5f8a")
    axis.set_xlabel("Drop in ROC-AUC when the column is shuffled")
    axis.set_title("Operational feature importance")
    figure.tight_layout()
    figure.savefig(destination / "feature_importance.png", dpi=120)
    plt.close(figure)


def _print_report(report: dict) -> None:
    target = report["target"]
    metrics = report["test_metrics"]
    print(f"Selected model: {report['selected_model']}")
    print(
        f"Long wait: {target['rule']} "
        f"(train positive rate {target['train_positive_rate']:.1%}, "
        f"test positive rate {target['test_positive_rate']:.1%})"
    )
    print("Training-fold ROC-AUC:")
    for name, scores in report["model_selection_roc_auc"].items():
        print(f"  {name}: {scores['mean_roc_auc']:.3f} ± {scores['std_roc_auc']:.3f}")
    print(
        "Held-out test: "
        f"ROC-AUC {metrics['roc_auc']:.3f}, "
        f"precision {metrics['precision']:.3f}, "
        f"recall {metrics['recall']:.3f}, "
        f"F1 {metrics['f1']:.3f}"
    )
    check = report["wait_component_check"]
    if check is not None:
        print(
            "Registration, triage, and clinician minutes track total_time "
            f"(correlation {check['correlation_with_total_time']:.3f}), "
            "so this agent leaves them out."
        )
    rate = target["test_positive_rate"]
    if rate < 0.05 or rate > 0.95:
        print(
            "The long-wait class is very unbalanced at this cutoff. "
            "Re-run with --threshold-minutes or --threshold-quantile 0.75 if the team wants a different definition."
        )
    print(f"Saved model and metrics under {Path(report['artifacts']['model']).parent}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ED/Operational long-wait agent.")
    parser.add_argument(
        "--data",
        required=True,
        help=f"Path to the cleaned ER wait-time CSV. Download it from {DATASET_URL}",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/operational",
        help="Directory for the model, metrics, and test predictions.",
    )
    parser.add_argument(
        "--threshold-minutes",
        type=float,
        default=DEFAULT_LONG_WAIT_MINUTES,
        help="A visit is long-wait when total_time is at least this many minutes.",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=None,
        help="Optional training-set quantile cutoff. Overrides --threshold-minutes.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "logistic_regression", "random_forest", "hist_gradient_boosting"],
        help="Classifier to fit. auto keeps the one with the best training ROC-AUC.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    visits = load_visits(args.data)
    report = train_operational_agent(
        visits,
        output_dir=args.output_dir,
        threshold_minutes=args.threshold_minutes,
        threshold_quantile=args.threshold_quantile,
        model_name=args.model,
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
