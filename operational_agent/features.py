"""Feature contract for the ED/Operational agent.

This agent predicts long-wait risk from facility and staffing context only.
Arrival timing and urgency belong to the triage agent. Identifiers, the wait
itself, and post-visit outcomes are not model inputs.
"""

from __future__ import annotations

# Shared with the rest of the team so every model is scored on the same patients.
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5

# Door-to-provider of 60 minutes. Total wait in this dataset runs from 4 to 442
# minutes. Change this if the team picks a different definition of "long."
DEFAULT_LONG_WAIT_MINUTES = 60.0

# Columns the trained model consumes, after nurse-ratio parsing.
CATEGORICAL_FEATURES = ["hospital_name", "region"]
NUMERIC_FEATURES = ["facility_size", "patients_per_nurse", "specialist_availability"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Raw dataset fields this agent is allowed to read. patients_per_nurse is
# derived from nurse_patient_ratio inside prepare_features.
SOURCE_FEATURES = [
    "hospital_name",
    "region",
    "facility_size",
    "nurse_patient_ratio",
    "specialist_availability",
]

# These are slices of total wait time (registration + triage + time to a
# clinician). Using them to predict total wait would leak the label.
LEAKAGE_COLUMNS = [
    "registration_time",
    "triage_time",
    "medprof_time",
    "total_time",
    "patient_outcome",
    "patient_satisfaction_score",
    "patient_satisfaction_category",
]

# Suggested split of the cleaned 22-column file. Share this with the team so
# two agents do not train on the same fields.
TEAM_FEATURE_SPLIT = {
    "operational_agent": SOURCE_FEATURES,
    "triage_agent": ["urgency", "time_day", "day_week", "season", "visit_date"],
    "patient_agent": (
        "This file has no age, sex, or complaint fields. "
        "patient_outcome and satisfaction are recorded after the visit."
    ),
    "target": "total_time, labeled long_wait when it meets the agreed minute cutoff",
    "do_not_use_as_inputs": LEAKAGE_COLUMNS
    + ["visit_id", "patient_id", "patient_id_text", "hospital_id", "hospital_id_text"],
}

# Clean-file names and the original Rivalytics names, keyed by stripped lowercase.
COLUMN_ALIASES = {
    "visit id": "visit_id",
    "patient id": "patient_id_text",
    "hospital id": "hospital_id_text",
    "patient_id": "patient_id",
    "hospital_id": "hospital_id",
    "hospital name": "hospital_name",
    "hospital_name": "hospital_name",
    "region": "region",
    "visit date": "visit_date",
    "visit_date": "visit_date",
    "day of week": "day_week",
    "day_week": "day_week",
    "season": "season",
    "time of day": "time_day",
    "time_day": "time_day",
    "urgency level": "urgency",
    "urgency": "urgency",
    "nurse-to-patient ratio": "nurse_patient_ratio",
    "nurse_to_patient_ratio": "nurse_patient_ratio",
    "nurse_patient_ratio": "nurse_patient_ratio",
    "specialist availability": "specialist_availability",
    "specialist_availability": "specialist_availability",
    "facility size (beds)": "facility_size",
    "facility_size": "facility_size",
    "time to registration (min)": "registration_time",
    "registration_time": "registration_time",
    "time to triage (min)": "triage_time",
    "triage_time": "triage_time",
    "time to medical professional (min)": "medprof_time",
    "medprof_time": "medprof_time",
    "total wait time (min)": "total_time",
    "total_time": "total_time",
    "patient outcome": "patient_outcome",
    "patient_outcome": "patient_outcome",
    "patient satisfaction": "patient_satisfaction_score",
    "patient_satisfaction": "patient_satisfaction_score",
    "patient_satisfaction_score": "patient_satisfaction_score",
    "patient_satisfaction_category": "patient_satisfaction_category",
}
