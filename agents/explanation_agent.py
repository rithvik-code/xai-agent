# Phase 5 Step 2 - NL Explanation Agent (Groq API)
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class ExplanationAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found! "
                "Add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        print("ExplanationAgent initialized ✅")

    def explain(self, scores, bias_results, compliance_results, domain="credit"):
        """Generate a plain-English explanation of the audit results"""

        total_score = scores["total"]["score"]
        grade = scores["total"]["grade"]
        fairness_score = bias_results["fairness_score"]
        compliance_score = compliance_results["compliance_score"]
        risk_tier = compliance_results["risk_tier"]["tier"]

        bias_summary = "\n".join([
            f"- {r['attribute']}: DPD={r['demographic_parity_difference']} "
            f"({r['dpd_severity']}), EOD={r['equalized_odds_difference']} "
            f"({r['eod_severity']})"
            for r in bias_results["bias_results"]
        ])

        proxy_summary = "\n".join([
            f"- {p['feature']} correlates with {p['protected_attribute']} "
            f"(correlation={p['correlation']}, {p['risk']})"
            for p in bias_results.get("proxy_risks", [])
        ]) or "None detected"

        prompt = f"""You are a responsible AI explainer. Translate this
technical XAI audit output into plain English for a non-technical
audience (e.g. a business executive or compliance officer).

Domain: {domain}
EU AI Act Risk Tier: {risk_tier}
Responsible AI Score: {total_score}/100 (Grade: {grade})
Fairness Score: {fairness_score}/100
Compliance Score: {compliance_score}/100

Bias findings:
{bias_summary}

Proxy features detected:
{proxy_summary}

Write a 2-paragraph plain-English summary. Paragraph 1 should explain
what the score means and whether this model is safe to deploy.
Paragraph 2 should explain the bias findings in simple terms and what
should be done about them. Avoid jargon. Be direct and clear."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        explanation = response.choices[0].message.content
        print("\n📝 PLAIN-ENGLISH EXPLANATION:")
        print("=" * 50)
        print(explanation)
        print("=" * 50)

        return explanation


# ── Run it! ──
if __name__ == "__main__":
    import sys
    import pandas as pd
    sys.path.append(".")

    from orchestrator.orchestrator_agent import OrchestratorAgent

    print("Loading data...")
    X_test = pd.read_csv("data/X_test.csv")
    X_train = pd.read_csv("data/X_train.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    orchestrator = OrchestratorAgent()
    orchestrator.load_all_agents(
        model_path="data/credit_model.pkl",
        feature_names_path="data/feature_names.pkl",
        X_train=X_train,
        rules_path="orchestrator/gdpr_rules.json"
    )

    report = orchestrator.run(
        task="Audit this credit model",
        X_test=X_test,
        y_test=y_test,
        domain="credit",
        num_samples=3
    )

    raw_scores = report["scores"]
    total_score = raw_scores["responsible_ai_score"]
    status = (
        "EXCELLENT" if total_score >= 90
        else "GOOD" if total_score >= 75
        else "NEEDS IMPROVEMENT" if total_score >= 60
        else "POOR"
    )
    scores = {
        "explainability": {"score": raw_scores["explainability"]},
        "fairness": {"score": raw_scores["fairness"]},
        "compliance": {"score": raw_scores["compliance"]},
        "total": {
            "score": total_score,
            "grade": raw_scores["grade"],
            "status": status
        }
    }

    agent = ExplanationAgent()
    agent.explain(
        scores=scores,
        bias_results=report["bias_results"],
        compliance_results=report["compliance_results"],
        domain="credit"
    )

    print("\n🎉 Phase 5 Step 2 Complete!")