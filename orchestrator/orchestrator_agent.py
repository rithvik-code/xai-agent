# Phase 3 - Orchestrator Agent
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(".")

from agents.interpretability_agent import InterpretabilityAgent
from agents.lime_agent import LimeAgent
from agents.bias_detector_agent import BiasDetectorAgent
from orchestrator.compliance_agent import ComplianceAgent

class OrchestratorAgent:
    def __init__(self):
        self.shap_agent = None
        self.lime_agent = None
        self.bias_agent = None
        self.compliance_agent = None
        self.all_results = {}
        print("OrchestratorAgent initialized ✅")

    def load_all_agents(self, model_path, feature_names_path,
                        X_train, rules_path):
        print("\nLoading all agents...")

        # SHAP Agent
        self.shap_agent = InterpretabilityAgent()
        self.shap_agent.load_model(model_path, feature_names_path)

        # LIME Agent
        self.lime_agent = LimeAgent()
        self.lime_agent.load_model(
            model_path, feature_names_path, X_train
        )

        # Bias Agent
        self.bias_agent = BiasDetectorAgent()
        self.bias_agent.load_model(model_path, feature_names_path)

        # Compliance Agent
        self.compliance_agent = ComplianceAgent()
        self.compliance_agent.load_rules(rules_path)

        print("\nAll agents loaded and ready! ✅")

    def compute_responsible_ai_score(self, bias_score,
                                      compliance_score,
                                      shap_works, lime_works):
        """Compute final Responsible AI Score 0-100"""
        score = 0

        # Explainability (30 points)
        if shap_works and lime_works:
            score += 30
        elif shap_works or lime_works:
            score += 15

        # Fairness (40 points)
        fairness_points = (bias_score / 100) * 40
        score += fairness_points

        # Compliance (30 points)
        compliance_points = (compliance_score / 100) * 30
        score += compliance_points

        return round(score)

    def get_grade(self, score):
        if score >= 90:
            return "A 🏆 EXCELLENT"
        elif score >= 75:
            return "B ✅ GOOD"
        elif score >= 60:
            return "C ⚠️ NEEDS WORK"
        elif score >= 40:
            return "D 🔴 POOR"
        else:
            return "F ❌ FAILING"

    def run(self, task, X_test, y_test,
            domain="credit", num_samples=3):
        """
        Main entry point - runs all agents
        and produces a unified report
        """
        print(f"\n{'='*60}")
        print(f"ORCHESTRATOR STARTING TASK:")
        print(f"'{task}'")
        print(f"{'='*60}")

        report = {
            "task": task,
            "domain": domain,
            "agents_run": []
        }

        # ── Step 1: Run SHAP ──
        print(f"\n{'='*60}")
        print("RUNNING AGENT 1: Interpretability (SHAP)")
        print(f"{'='*60}")
        shap_results = self.shap_agent.run(
            X_test, num_samples=num_samples
        )
        report["shap_results"] = shap_results
        report["agents_run"].append("SHAP ✅")
        shap_works = True
        print("SHAP Agent complete ✅")

        # ── Step 2: Run LIME ──
        print(f"\n{'='*60}")
        print("RUNNING AGENT 2: Interpretability (LIME)")
        print(f"{'='*60}")
        lime_results = self.lime_agent.run(
            X_test, num_samples=num_samples
        )
        report["lime_results"] = lime_results
        report["agents_run"].append("LIME ✅")
        lime_works = True
        print("LIME Agent complete ✅")

        # ── Step 3: Run Bias Detector ──
        print(f"\n{'='*60}")
        print("RUNNING AGENT 3: Bias Detector")
        print(f"{'='*60}")
        protected_cols = [
            col for col in X_test.columns
            if 'personal_status_sex' in col
        ]
        bias_results = self.bias_agent.run(
            X_test, y_test, protected_cols
        )
        report["bias_results"] = bias_results
        report["agents_run"].append("BiasDetector ✅")
        bias_score = bias_results["fairness_score"]
        print(f"Bias Agent complete ✅ Score: {bias_score}/100")

        # ── Step 4: Run Compliance ──
        print(f"\n{'='*60}")
        print("RUNNING AGENT 4: Compliance")
        print(f"{'='*60}")
        compliance_results = self.compliance_agent.run(
            domain=domain,
            has_shap=shap_works,
            has_lime=lime_works,
            bias_score=bias_score,
            feature_count=X_test.shape[1]
        )
        report["compliance_results"] = compliance_results
        report["agents_run"].append("Compliance ✅")
        compliance_score = compliance_results["compliance_score"]
        print(f"Compliance Agent complete ✅ Score: {compliance_score}/100")

        # ── Step 5: Compute Final Score ──
        responsible_ai_score = self.compute_responsible_ai_score(
            bias_score=bias_score,
            compliance_score=compliance_score,
            shap_works=shap_works,
            lime_works=lime_works
        )
        grade = self.get_grade(responsible_ai_score)

        report["scores"] = {
            "explainability": 30 if shap_works and lime_works else 15,
            "fairness": round((bias_score / 100) * 40),
            "compliance": round((compliance_score / 100) * 30),
            "responsible_ai_score": responsible_ai_score,
            "grade": grade
        }

        # ── Final Report ──
        print(f"\n{'='*60}")
        print("FINAL RESPONSIBLE AI REPORT")
        print(f"{'='*60}")
        print(f"Task: {task}")
        print(f"Domain: {domain}")
        print(f"\nAgents Run: {', '.join(report['agents_run'])}")
        print(f"\nSCORES:")
        print(f"  Explainability:      {report['scores']['explainability']}/30")
        print(f"  Fairness:            {report['scores']['fairness']}/40")
        print(f"  Compliance:          {report['scores']['compliance']}/30")
        print(f"  {'─'*35}")
        print(f"  RESPONSIBLE AI SCORE: {responsible_ai_score}/100")
        print(f"  GRADE: {grade}")
        print(f"\nAGENT AGREEMENT CHECK:")
        for i in range(num_samples):
            shap_pred = shap_results[i]["prediction"]
            lime_pred = lime_results[i]["prediction"]
            agree = "✅ AGREE" if shap_pred == lime_pred else "⚠️ DISAGREE"
            print(f"  Sample {i+1}: SHAP={shap_pred} LIME={lime_pred} → {agree}")

        print(f"\nTOP REMEDIATION STEPS:")
        for i, step in enumerate(
            compliance_results["remediation_steps"][:3]
        ):
            print(f"  {i+1}. {step['priority']}: {step['action']}")
        print(f"{'='*60}")

        return report


# ── Run it! ──
if __name__ == "__main__":
    print("Loading data...")
    X_test = pd.read_csv("data/X_test.csv")
    X_train = pd.read_csv("data/X_train.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    print("Data loaded ✅")

    # Create orchestrator
    orchestrator = OrchestratorAgent()
    orchestrator.load_all_agents(
        model_path="data/credit_model.pkl",
        feature_names_path="data/feature_names.pkl",
        X_train=X_train,
        rules_path="orchestrator/gdpr_rules.json"
    )

    # Run the full audit with one single call!
    report = orchestrator.run(
        task="Audit this credit model for GDPR compliance and gender bias",
        X_test=X_test,
        y_test=y_test,
        domain="credit",
        num_samples=3
    )

    print("\n" + "="*60)
    print("🎉 Phase 3 Complete!")
    print(f"Responsible AI Score: {report['scores']['responsible_ai_score']}/100")
    print(f"Grade: {report['scores']['grade']}")
    print("="*60)