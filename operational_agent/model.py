"""Train and score the ED/Operational long-wait classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from operational_agent.data import assert_no_leakage_in_features, prepare_features
from operational_agent.features import (
    CATEGORICAL_FEATURES,
    DECISION_THRESHOLD,
    NUMERIC_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def default_estimators(random_state: int) -> dict[str, object]:
    """Classifiers compared on the training fold before one is kept."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.08,
            max_iter=200,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


class OperationalAgent:
    """Facility and staffing model. ``predict_proba`` is the coordinator input."""

    def __init__(self, pipeline: Pipeline, model_name: str, threshold_minutes: float):
        self.pipeline = pipeline
        self.model_name = model_name
        self.threshold_minutes = threshold_minutes

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Probability of a long wait for each row, aligned with ``frame``."""
        features = prepare_features(frame)
        assert_no_leakage_in_features(features)
        return self.pipeline.predict_proba(features)[:, 1]

    def predict(self, frame: pd.DataFrame, decision_threshold: float = DECISION_THRESHOLD) -> np.ndarray:
        return (self.predict_proba(frame) >= decision_threshold).astype(int)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "model_name": self.model_name,
                "threshold_minutes": self.threshold_minutes,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "OperationalAgent":
        payload = joblib.load(path)
        return cls(
            pipeline=payload["pipeline"],
            model_name=payload["model_name"],
            threshold_minutes=float(payload["threshold_minutes"]),
        )


def select_model(
    features: pd.DataFrame,
    labels: pd.Series,
    estimators: dict[str, object],
    cv_folds: int,
    random_state: int,
) -> tuple[str, Pipeline, dict[str, dict[str, float]]]:
    """Pick the estimator with the highest mean ROC-AUC on training folds."""
    if labels.nunique() < 2:
        raise ValueError("The training label has only one class. Choose a different long-wait cutoff.")
    folds = min(cv_folds, int(labels.value_counts().min()))
    if folds < 2:
        raise ValueError("Not enough samples in each class for cross-validation.")
    folder = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores: dict[str, dict[str, float]] = {}
    best_name = ""
    best_mean = -1.0
    for name, estimator in estimators.items():
        pipeline = build_pipeline(estimator)
        fold_scores = cross_val_score(
            pipeline,
            features,
            labels,
            cv=folder,
            scoring="roc_auc",
            n_jobs=1,
        )
        mean_score = float(fold_scores.mean())
        scores[name] = {
            "mean_roc_auc": mean_score,
            "std_roc_auc": float(fold_scores.std()),
            "folds": int(folds),
        }
        if mean_score > best_mean:
            best_mean = mean_score
            best_name = name
    winner = build_pipeline(estimators[best_name])
    winner.fit(features, labels)
    return best_name, winner, scores


def classification_metrics(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    decision_threshold: float = DECISION_THRESHOLD,
) -> dict[str, float | list[list[int]]]:
    predicted = (probabilities >= decision_threshold).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "decision_threshold": float(decision_threshold),
        "positive_rate": float(np.mean(labels)),
        "confusion_matrix_labels_[0, 1]": matrix.tolist(),
    }
