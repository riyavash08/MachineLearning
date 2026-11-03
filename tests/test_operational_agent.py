"""Checks for feature parsing and a trainable operational agent."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from operational_agent.data import (
    long_wait_label,
    normalize_columns,
    parse_patients_per_nurse,
    prepare_features,
)
from operational_agent.features import LEAKAGE_COLUMNS, MODEL_FEATURES
from operational_agent.model import OperationalAgent
from operational_agent.train import train_operational_agent


def synthetic_visits(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hospitals = {
        "Springfield General Hospital": "Urban",
        "Riverside Medical Center": "Urban",
        "Summit Health Center": "Urban",
        "Northside Community Hospital": "Rural",
        "St. Mary's Regional Health": "Rural",
    }
    names = rng.choice(list(hospitals), rows)
    patients_per_nurse = rng.choice([2, 3, 4, 5], rows)
    facility_size = rng.integers(10, 201, rows)
    specialists = rng.integers(1, 11, rows)
    wait = (
        15
        + 22 * patients_per_nurse
        - 0.12 * facility_size
        - 2.0 * specialists
        + rng.normal(0, 6, rows)
    )
    wait = np.clip(wait, 4, 442)
    return pd.DataFrame(
        {
            "Visit ID": [f"V-{index:04d}" for index in range(rows)],
            "Patient ID": [f"PAT-{index:04d}" for index in range(rows)],
            "Hospital ID": ["HOSP-1"] * rows,
            "patient_id": np.arange(rows),
            "hospital_id": np.arange(rows) % 5,
            "hospital_name": names,
            "region": [hospitals[name] for name in names],
            "facility_size": facility_size,
            "nurse_patient_ratio": [f"1:{value}" for value in patients_per_nurse],
            "specialist_availability": specialists,
            "registration_time": wait * 0.2,
            "triage_time": wait * 0.3,
            "medprof_time": wait * 0.5,
            "total_time": wait,
            "urgency": rng.choice(["Low", "Medium", "High", "Critical"], rows),
        }
    )


class FeatureTests(unittest.TestCase):
    def test_ratio_formats(self):
        parsed = parse_patients_per_nurse(pd.Series(["1:4", "2:8", "1 to 5", "3", "4.5"]))
        self.assertTrue(
            np.allclose(parsed.to_numpy(dtype=float), [4, 4, 5, 3, 4.5])
        )

    def test_duplicate_id_columns_and_feature_order(self):
        raw = synthetic_visits(rows=20)
        normalized = normalize_columns(raw)
        self.assertIn("patient_id", normalized.columns)
        self.assertIn("patient_id_text", normalized.columns)
        features = prepare_features(raw)
        self.assertEqual(list(features.columns), MODEL_FEATURES)
        self.assertFalse(set(features.columns) & set(LEAKAGE_COLUMNS))

    def test_long_wait_cutoff_is_inclusive(self):
        labels = long_wait_label(pd.Series([59.9, 60, 61]), 60)
        self.assertEqual(labels.tolist(), [0, 1, 1])


class TrainingTests(unittest.TestCase):
    def test_agent_learns_synthetic_operational_signal(self):
        visits = normalize_columns(synthetic_visits())
        report = train_operational_agent(
            visits,
            output_dir="outputs/test_operational",
            threshold_minutes=60,
            cv_folds=2,
            estimators={
                "logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced")
            },
            model_name="logistic_regression",
        )
        self.assertGreater(report["test_metrics"]["roc_auc"], 0.8)
        self.assertEqual(report["selected_model"], "logistic_regression")
        self.assertAlmostEqual(report["wait_component_check"]["correlation_with_total_time"], 1.0, places=6)

        agent = OperationalAgent.load(report["artifacts"]["model"])
        probabilities = agent.predict_proba(visits.head(30))
        self.assertEqual(probabilities.shape, (30,))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        held_out = pd.read_csv(report["artifacts"]["test_predictions"])
        self.assertGreater(
            roc_auc_score(held_out["long_wait"], held_out["operational_probability"]),
            0.8,
        )


if __name__ == "__main__":
    unittest.main()
