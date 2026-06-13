# Step 5 - LIME Explanation Agent
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class LimeAgent:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        print("LimeAgent initialized ✅")

    def load_model(self, model_path, feature_names_path, X_train):
        # Load model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load feature names
        with open(feature_names_path, "rb") as f:
            self.feature_names = pickle.load(f)

        # Create LIME explainer using training data
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_names,
            class_names=["GOOD CREDIT", "BAD CREDIT"],
            mode="classification",
            random_state=42
        )
        print(f"LIME explainer ready! ✅")

    def explain_prediction(self, data_row, row_index=0):
        """Explain a single prediction using LIME"""
        if isinstance(data_row, pd.DataFrame):
            row_values = data_row.values[0]
        else:
            row_values = data_row

        # Get prediction
        prediction = self.model.predict(data_row)[0]
        probability = self.model.predict_proba(data_row)[0]
        label = "BAD CREDIT RISK" if prediction == 1 else "GOOD CREDIT RISK"
        confidence = probability[prediction] * 100

        print(f"\n{'='*50}")
        print(f"LIME PREDICTION: {label}")
        print(f"CONFIDENCE: {confidence:.1f}%")
        print(f"{'='*50}")

        # Generate LIME explanation
        explanation = self.explainer.explain_instance(
            data_row=row_values,
            predict_fn=self.model.predict_proba,
            num_features=5
        )

        # Show top reasons
        print(f"\nTOP 5 LIME REASONS:")
        print(f"{'-'*50}")
        for i, (feat, impact) in enumerate(explanation.as_list()):
            direction = "INCREASES RISK 📈" if impact > 0 else "DECREASES RISK 📉"
            print(f"{i+1}. {feat}: {impact:.4f} → {direction}")

        # Save chart
        os.makedirs("reports", exist_ok=True)
        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation - Row {row_index}: {label}")
        plt.tight_layout()
        chart_path = f"reports/lime_explanation_row{row_index}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Chart saved to {chart_path} ✅")

        return {
            "prediction": int(prediction),
            "label": label,
            "confidence": round(confidence, 2),
            "top_reasons": explanation.as_list(),
            "chart_path": chart_path
        }

    def run(self, X_test, num_samples=3):
        """Run LIME on multiple samples"""
        print(f"\nRunning LIME on {num_samples} samples...")
        results = []
        for i in range(num_samples):
            row = X_test.iloc[[i]]
            result = self.explain_prediction(row, row_index=i)
            results.append(result)
        return results

    def compare_with_shap(self, shap_results, lime_results):
        """Compare SHAP vs LIME top features"""
        print(f"\n{'='*50}")
        print("SHAP vs LIME COMPARISON")
        print(f"{'='*50}")
        for i in range(len(shap_results)):
            print(f"\nSample {i+1}:")
            print(f"  SHAP top feature: {shap_results[i]['top_reasons'][0][0]}")
            print(f"  LIME top feature: {lime_results[i]['top_reasons'][0][0]}")
            agree = shap_results[i]['prediction'] == lime_results[i]['prediction']
            print(f"  Both agree on prediction: {'YES ✅' if agree else 'NO ⚠️'}")


# ── Run it! ──
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_test = pd.read_csv("data/X_test.csv")
    X_train = pd.read_csv("data/X_train.csv")
    print(f"Data loaded ✅")

    # Run LIME agent
    agent = LimeAgent()
    agent.load_model("data/credit_model.pkl", "data/feature_names.pkl", X_train)
    lime_results = agent.run(X_test, num_samples=3)

    # Compare with SHAP
    import sys
    sys.path.append(".")
    from agents.interpretability_agent import InterpretabilityAgent
    shap_agent = InterpretabilityAgent()
    shap_agent.load_model("data/credit_model.pkl", "data/feature_names.pkl")
    shap_results = shap_agent.run(X_test, num_samples=3)

    agent.compare_with_shap(shap_results, lime_results)

    print("\n" + "="*50)
    print("Step 5 Complete! LIME Agent is working!")
    print("Check reports/ folder for LIME charts!")
    print("="*50)