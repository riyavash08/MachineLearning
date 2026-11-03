"""Load the ER wait-time table and build the operational feature frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from operational_agent.features import (
    COLUMN_ALIASES,
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    SOURCE_FEATURES,
)

DATASET_URL = "https://www.kaggle.com/datasets/yasmeenkotifani/er-wait-time-dataset-clean"
CSV_NAME = "ER Wait Time Dataset - clean.csv"


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename original and cleaned headers onto one set of snake_case names."""
    renamed = frame.copy()
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for column in renamed.columns:
        key = str(column).strip().lower()
        canonical = COLUMN_ALIASES.get(key, key.replace(" ", "_"))
        if canonical in used:
            canonical = f"{canonical}__dup"
        used.add(canonical)
        mapping[column] = canonical
    return renamed.rename(columns=mapping)


def parse_patients_per_nurse(values: pd.Series) -> pd.Series:
    """Convert a nurse-to-patient ratio into patients per nurse.

    Accepts ``1:4``, ``1 to 4``, ``2:8`` (also 4 patients per nurse), and a
    plain number. A plain number is already treated as patients per nurse,
    which is how a cleaned numeric column is stored.
    """
    text = values.astype("string").str.strip()
    parts = text.str.extract(r"(?i)(\d+(?:\.\d+)?)\s*(?::|/|to)\s*(\d+(?:\.\d+)?)")
    nurses = pd.to_numeric(parts[0], errors="coerce")
    patients = pd.to_numeric(parts[1], errors="coerce")
    from_ratio = patients / nurses.replace(0, np.nan)
    plain = pd.to_numeric(text, errors="coerce")
    return from_ratio.fillna(plain)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the five operational columns, in model order.

    ``frame`` may be the raw CSV or a frame that was already normalized.
    """
    data = frame if _is_normalized(frame) else normalize_columns(frame)
    missing = [name for name in SOURCE_FEATURES if name not in data.columns and name != "hospital_name"]
    if "hospital_name" not in data.columns and "hospital_id" not in data.columns and "hospital_id_text" not in data.columns:
        missing.append("hospital_name")
    if missing:
        found = ", ".join(map(str, data.columns))
        needed = ", ".join(missing)
        raise ValueError(f"Operational agent is missing columns: {needed}. Found: {found}")

    features = pd.DataFrame(index=data.index)
    if "hospital_name" in data.columns:
        features["hospital_name"] = data["hospital_name"].astype("string").str.strip()
    else:
        source = "hospital_id" if "hospital_id" in data.columns else "hospital_id_text"
        features["hospital_name"] = data[source].astype("string").str.strip()
    features["region"] = data["region"].astype("string").str.strip()
    features["facility_size"] = pd.to_numeric(data["facility_size"], errors="coerce")
    features["patients_per_nurse"] = parse_patients_per_nurse(data["nurse_patient_ratio"])
    features["specialist_availability"] = pd.to_numeric(data["specialist_availability"], errors="coerce")
    return features[MODEL_FEATURES]


def long_wait_label(total_time: pd.Series, threshold_minutes: float) -> pd.Series:
    """1 when total wait is at least ``threshold_minutes``, else 0."""
    minutes = pd.to_numeric(total_time, errors="coerce")
    if minutes.isna().any():
        raise ValueError("total_time has values that are not numeric.")
    return (minutes >= threshold_minutes).astype(int)


def load_visits(path: str | Path) -> pd.DataFrame:
    """Read the wait-time CSV and normalize its column names."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Download '{CSV_NAME}' from {DATASET_URL} "
            "and pass the file path to the trainer."
        )
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"{csv_path} has no rows.")
    return normalize_columns(frame)


def component_leakage_report(frame: pd.DataFrame) -> dict[str, float] | None:
    """Show how strongly the wait components reconstruct total wait time."""
    parts = ["registration_time", "triage_time", "medprof_time"]
    if "total_time" not in frame.columns or any(name not in frame.columns for name in parts):
        return None
    numeric = frame[parts + ["total_time"]].apply(pd.to_numeric, errors="coerce")
    component_sum = numeric[parts].sum(axis=1)
    return {
        "correlation_with_total_time": float(component_sum.corr(numeric["total_time"])),
        "median_absolute_difference_minutes": float((component_sum - numeric["total_time"]).abs().median()),
    }


def assert_no_leakage_in_features(features: pd.DataFrame) -> None:
    overlap = sorted(set(features.columns) & set(LEAKAGE_COLUMNS))
    if overlap:
        raise ValueError(f"Operational features include leakage columns: {overlap}")


def _is_normalized(frame: pd.DataFrame) -> bool:
    return "nurse_patient_ratio" in frame.columns or "total_time" in frame.columns
