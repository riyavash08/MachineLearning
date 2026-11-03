
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "total_time"

# Define "long wait". Option A: fixed threshold in minutes.
# Option B (default): top 25% of wait times in the TRAINING set.
LONG_WAIT_MINUTES = 60      # e.g. 120 to use a fixed cutoff
#LONG_WAIT_QUANTILE = 0.75     # used only if LONG_WAIT_MINUTES is None

# Triage / arrival-related features (keep mutually exclusive from the other agents' feature sets).
TRIAGE_CATEGORICAL = ["urgency", "day_week", "time_day", "season"]
TRIAGE_NUMERIC = []           # add any numeric triage/arrival columns here

RANDOM_STATE = 42

class TriageAgent:

    name = "triage"

    def __init__(self, categorical=None, numeric=None, random_state=RANDOM_STATE):
        self.categorical = categorical if categorical is not None else TRIAGE_CATEGORICAL
        self.numeric = numeric if numeric is not None else TRIAGE_NUMERIC
        self.features = self.categorical + self.numeric

        transformers = [
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical)
        ]
        if self.numeric:
            transformers.append(("num", StandardScaler(), self.numeric))

        self.model = Pipeline([
            ("prep", ColumnTransformer(transformers)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced",   # long waits are the minority class
                random_state=random_state,
                n_jobs=-1,
            )),
        ])

    def fit(self, X: pd.DataFrame, y):
        self.model.fit(X[self.features], y)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(long wait) per patient. This is what the Coordinator consumes."""
        return self.model.predict_proba(X[self.features])[:, 1]

    def predict(self, X: pd.DataFrame, threshold=0.5) -> np.ndarray:
        return (self.predict_risk(X) >= threshold).astype(int)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_long_wait_label(wait, cutoff):
    return (wait >= cutoff).astype(int)


def main(csv_path):
    df = pd.read_csv(csv_path)

    missing = [c for c in [TARGET_COL] + TRIAGE_CATEGORICAL + TRIAGE_NUMERIC
               if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}\n"
                         f"Available columns: {list(df.columns)}")

    # Split first, then compute the long-wait cutoff from training data only
    # (avoids leaking test-set information into the label definition).
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE
    )
    cutoff = (LONG_WAIT_MINUTES if LONG_WAIT_MINUTES is not None
              else train_df[TARGET_COL].quantile(LONG_WAIT_QUANTILE))
    print(f"Long-wait cutoff: {cutoff:.1f} minutes")

    y_train = make_long_wait_label(train_df[TARGET_COL], cutoff)
    y_test = make_long_wait_label(test_df[TARGET_COL], cutoff)
    print(f"Long-wait rate  train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

    agent = TriageAgent().fit(train_df, y_train)

    risk = agent.predict_risk(test_df)
    pred = (risk >= 0.5).astype(int)

    print("\n=== Triage Agent: test results ===")
    print(f"ROC AUC:           {roc_auc_score(y_test, risk):.3f}")
    print(f"Average precision: {average_precision_score(y_test, risk):.3f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\n", classification_report(y_test, pred,
                                      target_names=["Not long wait", "Long wait"]))

    # Save per-patient risk scores so the Coordinator can load them later.
    out = pd.DataFrame({"index": test_df.index, "triage_risk": risk,
                        "long_wait": y_test.values})
    out.to_csv("triage_agent_predictions.csv", index=False)
    print("Saved predictions to triage_agent_predictions.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python triage_agent.py path/to/ed_wait_times.csv")
        sys.exit(1)
    main(sys.argv[1])