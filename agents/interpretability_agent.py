# Step 4 - Interpretability Agent using SHAP
import shap
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # prevents popup windows
import matplotlib.pyplot as plt
import os

class InterpretabilityAgent:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        print("InterpretabilityAgent initialized ✅")

    def load_model(self, model_path, feature_names_path):
        # Load the trained model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load feature names
        with open(feature_names_path, "rb") as f:
            self.feature_names = pickle.load(f)

        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        print(f"Model loaded! Features: {len(self.feature_names)} ✅")

    def explain_prediction(self, data_row, row_index=0):
        """Explain a single prediction"""
        # Make sure it's the right shape
        if isinstance(data_row, pd.Series):
            data_row = data_row.to_frame().T

        # Get prediction
        prediction = self.model.predict(data_row)[0]
        probability = self.model.predict_proba(data_row)[0]

        label = "BAD CREDIT RISK 🔴" if prediction == 1 else "GOOD CREDIT RISK 🟢"
        confidence = probability[prediction] * 100

        print(f"\n{'='*50}")
        print(f"PREDICTION: {label}")
        print(f"CONFIDENCE: {confidence:.1f}%")
        print(f"{'='*50}")

        # Get SHAP values
        shap_values = self.explainer.shap_values(data_row)

        # For binary classification, use class 1 (bad credit) values
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        # Show top 5 reasons
        sv_flat = np.array(sv).flatten()
        feature_impacts = list(zip(self.feature_names, sv_flat))
        feature_impacts.sort(key=lambda x: float(abs(x[1])), reverse=True)

        print(f"\nTOP 5 REASONS FOR THIS DECISION:")
        print(f"{'-'*50}")
        for i, (feat, impact) in enumerate(feature_impacts[:5]):
            direction = "INCREASES RISK 📈" if impact > 0 else "DECREASES RISK 📉"
            print(f"{i+1}. {feat}: {impact:.4f} → {direction}")

        # Save a chart
        os.makedirs("reports", exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        top_features = [f[0] for f in feature_impacts[:10]]
        top_values = [f[1] for f in feature_impacts[:10]]
        colors = ['#ff6b6b' if v > 0 else '#51cf66' for v in top_values]
        
        plt.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
        plt.xlabel('SHAP Value (Impact on prediction)')
        plt.title(f'Why did the AI decide: {label}\nRow {row_index}')
        plt.tight_layout()
        
        chart_path = f"reports/shap_explanation_row{row_index}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nChart saved to {chart_path} ✅")

        return {
            "prediction": int(prediction),
            "label": label,
            "confidence": round(confidence, 2),
            "top_reasons": feature_impacts[:5],
            "chart_path": chart_path
        }

    def run(self, X_test, num_samples=3):
        """Run explanations on multiple samples"""
        print(f"\nRunning explanations on {num_samples} samples...")
        results = []
        for i in range(num_samples):
            row = X_test.iloc[[i]]
            result = self.explain_prediction(row, row_index=i)
            results.append(result)
        return results


# ── Run it! ──
if __name__ == "__main__":
    # Load test data
    print("Loading data...")
    X_test = pd.read_csv("data/X_test.csv")
    print(f"Test data loaded: {X_test.shape} ✅")

    # Create and run agent
    agent = InterpretabilityAgent()
    agent.load_model("data/credit_model.pkl", "data/feature_names.pkl")

    # Explain 3 predictions
    results = agent.run(X_test, num_samples=3)

    print("\n" + "="*50)
    print("🎉 Step 4 Complete! SHAP Agent is working!")
    print(f"Charts saved in the reports/ folder")
    print("="*50)