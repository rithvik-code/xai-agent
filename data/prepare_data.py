# prepare_data.py - 79.5% version
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
import os

# ── 1. Download dataset ──
print("Downloading dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

columns = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment", "housing", "existing_credits", "job",
    "dependents", "telephone", "foreign_worker", "target"
]

df = pd.read_csv(url, sep=" ", header=None, names=columns)
print(f"Dataset loaded! Shape: {df.shape} ✅")

# ── 2. Prepare features ──
df["target"] = df["target"].map({1: 0, 2: 1})
df = pd.get_dummies(df, drop_first=True)

# ── 3. Split ──
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)} ✅")

# ── 4. Balance with SMOTE ──
print("\nBalancing training data...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"Good credit: {(y_train_bal==0).sum()}")
print(f"Bad credit:  {(y_train_bal==1).sum()} ✅")

# ── 5. Train XGBoost ──
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=2,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_bal, y_train_bal)

# ── 6. Check accuracy ──
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy:.2%}")

# ── 7. Save everything ──
os.makedirs("data", exist_ok=True)

with open("data/credit_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved ✅")

X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
X_train.to_csv("data/X_train.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
print("Data saved ✅")

feature_names = list(X_train.columns)
with open("data/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print("Feature names saved ✅")

print("\n🎉 Complete! Your model is ready!")