
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# 1. Load data and define target
df = pd.read_csv("ER Wait Time Dataset - clean.csv")
df["Long_Wait"] = (df["Total_Time"] >= 60).astype(int)

# 2. Use Urgency as the prototype feature
X = df[["Urgency"]]
y = df["Long_Wait"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Train Patient Agent
patient_agent = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )
)

patient_agent.fit(X_train, y_train)

# 5. Evaluate Patient Agent
y_pred = patient_agent.predict(X_test)
y_prob = patient_agent.predict_proba(X_test)[:, 1]

print("\n=== PATIENT AGENT ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# 6. Compare against a simple baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("\n=== MAJORITY-CLASS BASELINE ===")
print("Accuracy:", round(accuracy_score(y_test, baseline_pred), 4))
print("F1-score:", round(f1_score(y_test, baseline_pred), 4))

# 7. Export probabilities for the coordinator
patient_output = pd.DataFrame({
    "row_index": X_test.index,
    "Patient_Agent_Probability": y_prob,
    "Actual_Long_Wait": y_test.to_numpy()
})

patient_output.to_csv("patient_agent_predictions.csv", index=False)
joblib.dump(patient_agent, "patient_agent_model.pkl")

print("\nSaved patient_agent_predictions.csv")
print("Saved patient_agent_model.pkl")

# 8. Analyze Urgency and Long-Wait Relationship

urgency_analysis = df.groupby("Urgency").agg(
    Total_Visits=("Long_Wait", "count"),
    Long_Wait_Count=("Long_Wait", "sum"),
    Average_Wait=("Total_Time", "mean"),
    Long_Wait_Rate=("Long_Wait", "mean")
)

urgency_analysis["Long_Wait_Rate"] = (
    urgency_analysis["Long_Wait_Rate"] * 100
).round(2)

urgency_analysis["Average_Wait"] = (
    urgency_analysis["Average_Wait"]
).round(2)

print("\n=== URGENCY ANALYSIS ===")
print(urgency_analysis)

print(urgency_analysis.to_string())

#9 Building The Urgency-Rate Baseline
#Calculate long-wait rates using training data only
train_rates = pd.DataFrame({
    "Urgency": X_train["Urgency"],
    "Long_Wait": y_train
}).groupby("Urgency")["Long_Wait"].mean()

print("\n=== TRAINING LONG-WAIT RATES ===")
print((train_rates * 100).round(2))


#Assign probabilities to test visits
baseline_prob = X_test["Urgency"].map(train_rates).to_numpy()


#Convert probabilities into predicted classes
baseline_pred = (baseline_prob >= 0.5).astype(int)


#Evaluate the baseline
print("\n=== URGENCY-RATE BASELINE ===")

print("Accuracy:",
      round(accuracy_score(y_test, baseline_pred), 4))

print("Precision:",
      round(precision_score(y_test, baseline_pred, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_test, baseline_pred, zero_division=0), 4))

print("F1-score:",
      round(f1_score(y_test, baseline_pred, zero_division=0), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, baseline_prob), 4))


#Compare probability reliability
from sklearn.metrics import brier_score_loss

print("\n=== BRIER SCORE COMPARISON ===")

print("Random Forest:",
      round(brier_score_loss(y_test, y_prob), 4))

print("Urgency-Rate Baseline:",
      round(brier_score_loss(y_test, baseline_prob), 4))


# Compare individual predicted probabilities
comparison = pd.DataFrame({
    "Urgency": X_test["Urgency"].to_numpy(),
    "Random_Forest_Probability": y_prob,
    "Urgency_Baseline_Probability": baseline_prob
})

comparison["Probability_Difference"] = (
    comparison["Random_Forest_Probability"]
    - comparison["Urgency_Baseline_Probability"]
).abs()

print("\n=== PROBABILITY COMPARISON ===")

print("Average difference:",
      round(comparison["Probability_Difference"].mean(), 6))

print("Maximum difference:",
      round(comparison["Probability_Difference"].max(), 6))

print("\nAverage probabilities by urgency:")

print(
    comparison.groupby("Urgency")[
        ["Random_Forest_Probability", "Urgency_Baseline_Probability"]
    ].mean().round(4)
)

print("\n=== ALL AVAILABLE FEATURES ===")

for i, column in enumerate(df.columns, 1):
    print(i, column)

print("\n=== PATIENT-RELATED FEATURES ===")

patient_columns = [
    "Patient ID",
    "Patient_ID",
    "Urgency",
    "Patient_Outcome",
    "Patient_Satisfaction_Score",
    "Patient_Satisfaction_Category"
]

for column in patient_columns:
    if column in df.columns:
        print("\n", column)
        print(df[column].value_counts().head(10))

print("Are Patient ID columns identical?")
print(df["Patient ID"].equals(df["Patient_ID"]))

print("\nNumber of unique patients:")
print(df["Patient ID"].nunique())

patient_visits = df.groupby("Patient ID").size()

print("\nPatients with multiple visits:")
print((patient_visits > 1).sum())

print("\nMaximum visits by one patient:")
print(patient_visits.max())