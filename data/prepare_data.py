# Step 3 - Download dataset and train first model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# ── 1. Load the German Credit dataset directly from the web ──
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
print(f"Dataset loaded! Shape: {df.shape}")
print(df.head())

# ── 2. Prepare features ──
# target: 1 = good credit, 2 = bad credit → convert to 0 and 1
df["target"] = df["target"].map({1: 0, 2: 1})

# Convert all categorical columns to numbers
df = pd.get_dummies(df, drop_first=True)

X = df.drop("target", axis=1)
y = df["target"]

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ── 3. Split into train and test ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ── 4. Train a Random Forest model ──
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── 5. Check accuracy ──
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2%}")

# ── 6. Save everything ──
os.makedirs("data", exist_ok=True)

# Save model
with open("data/credit_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved to data/credit_model.pkl ✅")

# Save test data for later use by agents
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
X_train.to_csv("data/X_train.csv", index=False)
print("Test data saved to data/ folder ✅")

# Save feature names (needed for SHAP later)
feature_names = list(X.columns)
with open("data/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print("Feature names saved ✅")

print("\n🎉 Step 3 complete! Your first model is trained and saved!")