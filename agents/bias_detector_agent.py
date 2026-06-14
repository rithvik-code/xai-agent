# Phase 2 - Bias Detector Agent
import pandas as pd
import numpy as np
import pickle
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate
)
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# aif360 imports
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric
    AIF360_AVAILABLE = True
    print("aif360 loaded ✅")
except ImportError:
    AIF360_AVAILABLE = False
    print("aif360 not available, skipping those checks")


class BiasDetectorAgent:
    def __init__(self):
        self.model = None
        self.feature_names = None
        print("BiasDetectorAgent initialized ✅")

    def load_model(self, model_path, feature_names_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(feature_names_path, "rb") as f:
            self.feature_names = pickle.load(f)
        print("Model loaded into BiasDetectorAgent ✅")

    def get_severity(self, value):
        value = abs(value)
        if value < 0.05:
            return "PASS ✅"
        elif value < 0.10:
            return "WARN ⚠️"
        else:
            return "FAIL ❌"

    def check_protected_attribute(self, X_test, y_test, predictions,
                                   attr_col, attr_name):
        print(f"\n{'='*50}")
        print(f"CHECKING BIAS FOR: {attr_name}")
        print(f"{'='*50}")

        protected = X_test[attr_col]
        unique_groups = protected.unique()
        print(f"\nGroups found: {unique_groups}")

        print(f"\nPREDICTION RATES PER GROUP:")
        print(f"{'-'*40}")
        for group in sorted(unique_groups):
            mask = protected == group
            group_preds = predictions[mask]
            rate = group_preds.mean() * 100
            count = mask.sum()
            print(f"  Group {group}: {rate:.1f}% flagged as bad credit "
                  f"({count} people)")

        dpd = demographic_parity_difference(
            y_true=y_test,
            y_pred=predictions,
            sensitive_features=protected
        )

        eod = equalized_odds_difference(
            y_true=y_test,
            y_pred=predictions,
            sensitive_features=protected
        )

        metric_frame = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "selection_rate": selection_rate
            },
            y_true=y_test,
            y_pred=predictions,
            sensitive_features=protected
        )

        print(f"\nFAIRNESS METRICS:")
        print(f"{'-'*40}")
        print(f"Demographic Parity Difference: "
              f"{dpd:.4f} → {self.get_severity(dpd)}")
        print(f"Equalized Odds Difference:     "
              f"{eod:.4f} → {self.get_severity(eod)}")
        print(f"\nACCURACY PER GROUP:")
        print(metric_frame.by_group)

        return {
            "attribute": attr_name,
            "demographic_parity_difference": round(float(dpd), 4),
            "equalized_odds_difference": round(float(eod), 4),
            "dpd_severity": self.get_severity(dpd),
            "eod_severity": self.get_severity(eod),
            "by_group": metric_frame.by_group.to_dict()
        }

    def check_proxy_features(self, X_test, protected_cols):
        print(f"\n{'='*50}")
        print("CHECKING FOR PROXY FEATURES")
        print(f"{'='*50}")

        proxy_risks = []
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns

        for protected_col in protected_cols:
            if protected_col not in X_test.columns:
                continue
            protected = X_test[protected_col]
            print(f"\nFeatures correlated with {protected_col}:")
            print(f"{'-'*40}")
            found_any = False
            for col in numeric_cols:
                if col == protected_col:
                    continue
                corr = abs(X_test[col].corr(protected))
                if corr > 0.1:
                    found_any = True
                    risk = (
                        "HIGH RISK 🔴" if corr > 0.3
                        else "MEDIUM RISK 🟡" if corr > 0.2
                        else "LOW RISK 🟢"
                    )
                    print(f"  {col}: correlation={corr:.3f} → {risk}")
                    proxy_risks.append({
                        "feature": col,
                        "protected_attribute": protected_col,
                        "correlation": round(corr, 3),
                        "risk": risk
                    })
            if not found_any:
                print("  No proxy features found for this attribute")

        if not proxy_risks:
            print("\n  No strong proxy features found ✅")

        return proxy_risks

    def check_aif360(self, X_test, y_test, predictions, protected_col):
        """Extra bias checks using IBM's aif360"""
        if not AIF360_AVAILABLE:
            return None
        try:
            print(f"\n{'='*50}")
            print(f"AIF360 CHECKS FOR: {protected_col}")
            print(f"{'='*50}")

            df_aif = X_test.copy()
            df_aif['target'] = y_test.values

            dataset_orig = BinaryLabelDataset(
                df=df_aif[list(X_test.columns) + ['target']],
                label_names=['target'],
                protected_attribute_names=[protected_col],
                favorable_label=0,
                unfavorable_label=1
            )

            privileged = [{protected_col: 1.0}]
            unprivileged = [{protected_col: 0.0}]

            metric = BinaryLabelDatasetMetric(
                dataset_orig,
                unprivileged_groups=unprivileged,
                privileged_groups=privileged
            )

            di = metric.disparate_impact()
            spd = metric.statistical_parity_difference()

            print(f"Disparate Impact:              {di:.4f}")
            print(f"  → {'PASS ✅' if 0.8 <= di <= 1.2 else 'FAIL ❌'}")
            print(f"  (Fair range: 0.8 to 1.2)")
            print(f"Statistical Parity Difference: {spd:.4f}")
            print(f"  → {'PASS ✅' if abs(spd) < 0.1 else 'FAIL ❌'}")

            return {
                "disparate_impact": round(float(di), 4),
                "statistical_parity_difference": round(float(spd), 4),
                "di_fair": 0.8 <= di <= 1.2,
                "spd_fair": abs(spd) < 0.1
            }

        except Exception as e:
            print(f"aif360 check skipped: {e}")
            return None

    def compute_fairness_score(self, results):
        score = 100
        for r in results:
            if "FAIL" in r["dpd_severity"]:
                score -= 20
            elif "WARN" in r["dpd_severity"]:
                score -= 10
            if "FAIL" in r["eod_severity"]:
                score -= 20
            elif "WARN" in r["eod_severity"]:
                score -= 10
        return max(0, score)

    def run(self, X_test, y_test, protected_cols):
        print("\n🔍 STARTING BIAS DETECTION AUDIT...")

        predictions = self.model.predict(X_test)

        results = []
        for col in protected_cols:
            if col in X_test.columns:
                result = self.check_protected_attribute(
                    X_test, y_test, predictions, col, col
                )
                aif_result = self.check_aif360(
                    X_test, y_test, predictions, col
                )
                if aif_result:
                    result["aif360"] = aif_result
                results.append(result)

        proxy_risks = self.check_proxy_features(X_test, protected_cols)
        fairness_score = self.compute_fairness_score(results)

        print(f"\n{'='*50}")
        print(f"OVERALL FAIRNESS SCORE: {fairness_score}/100")
        if fairness_score >= 80:
            print("Rating: FAIR ✅")
        elif fairness_score >= 60:
            print("Rating: NEEDS IMPROVEMENT ⚠️")
        else:
            print("Rating: BIASED ❌")
        print(f"{'='*50}")

        return {
            "bias_results": results,
            "proxy_risks": proxy_risks,
            "fairness_score": fairness_score,
            "predictions": predictions.tolist()
        }


# ── Run it! ──
if __name__ == "__main__":
    print("Loading data...")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    print(f"Data loaded ✅")

    protected_cols = [
        col for col in X_test.columns
        if 'personal_status_sex' in col
    ]
    print(f"\nProtected attributes found: {protected_cols}")

    agent = BiasDetectorAgent()
    agent.load_model("data/credit_model.pkl", "data/feature_names.pkl")
    results = agent.run(X_test, y_test, protected_cols)

    print("\n" + "="*50)
    print("Phase 2 Complete!")
    print(f"Fairness Score: {results['fairness_score']}/100")
    print("="*50)