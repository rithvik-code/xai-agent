# Responsible AI Score Module
import json
import os
from datetime import datetime


def compute_responsible_ai_score(bias_score, compliance_score,
                                  shap_works, lime_works,
                                  proxy_risks, gdpr_results):
    scores = {}

    # ── 1. Explainability (30 points) ──
    exp_score = 0
    if shap_works and lime_works:
        exp_score = 30
    elif shap_works or lime_works:
        exp_score = 15
    scores["explainability"] = {
        "score": exp_score,
        "max": 30,
        "shap": shap_works,
        "lime": lime_works,
        "status": "EXCELLENT ✅" if exp_score == 30
                  else "PARTIAL ⚠️" if exp_score > 0
                  else "FAILING ❌"
    }

    # ── 2. Fairness (40 points) ──
    # 20 points for detecting and measuring bias
    # 10 points for proxy feature detection
    # 10 points based on actual fairness score
    detection_score = 20
    proxy_score = 10
    actual_fairness = round((bias_score / 100) * 10)
    fair_score = min(40, detection_score + proxy_score + actual_fairness)
    scores["fairness"] = {
        "score": fair_score,
        "max": 40,
        "bias_score": bias_score,
        "proxy_risks_found": len(proxy_risks),
        "status": "EXCELLENT ✅" if fair_score >= 32
                  else "GOOD ✅" if fair_score >= 25
                  else "NEEDS WORK ⚠️" if fair_score >= 20
                  else "FAILING ❌"
    }

    # ── 3. Compliance (30 points) ──
    comp_score = round((compliance_score / 100) * 30)
    scores["compliance"] = {
        "score": comp_score,
        "max": 30,
        "compliance_score": compliance_score,
        "status": "EXCELLENT ✅" if comp_score >= 24
                  else "GOOD ✅" if comp_score >= 20
                  else "NEEDS WORK ⚠️" if comp_score >= 15
                  else "FAILING ❌"
    }

    # ── 4. Total ──
    total = exp_score + fair_score + comp_score
    scores["total"] = {
        "score": total,
        "max": 100,
        "grade": get_grade(total),
        "status": get_status(total)
    }

    return scores


def get_grade(score):
    if score >= 90: return "A 🏆 EXCELLENT"
    elif score >= 75: return "B ✅ GOOD"
    elif score >= 60: return "C ⚠️ NEEDS WORK"
    elif score >= 40: return "D 🔴 POOR"
    else: return "F ❌ FAILING"


def get_status(score):
    if score >= 90: return "READY FOR DEPLOYMENT ✅"
    elif score >= 75: return "NEARLY READY ⚠️"
    elif score >= 60: return "NEEDS IMPROVEMENT 🔧"
    else: return "NOT READY FOR DEPLOYMENT ❌"


def save_audit_history(report, save_path="data/audit_history.json"):
    """Save every audit to a history file"""
    history = []

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            history = json.load(f)

    report["timestamp"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    report["audit_number"] = len(history) + 1

    history.append({
        "audit_number": report["audit_number"],
        "timestamp": report["timestamp"],
        "task": report.get("task", "Unknown"),
        "domain": report.get("domain", "Unknown"),
        "responsible_ai_score": report.get(
            "scores", {}
        ).get("total", {}).get("score", 0),
        "grade": report.get(
            "scores", {}
        ).get("total", {}).get("grade", "Unknown"),
        "bias_score": report.get(
            "bias_results", {}
        ).get("fairness_score", 0),
        "compliance_score": report.get(
            "compliance_results", {}
        ).get("compliance_score", 0)
    })

    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nAudit #{report['audit_number']} saved to history ✅")
    return history


def print_score_card(scores):
    print(f"\n{'='*60}")
    print(f"  RESPONSIBLE AI SCORE CARD")
    print(f"{'='*60}")
    print(f"  Explainability:  {scores['explainability']['score']:>3}/30  "
          f"{scores['explainability']['status']}")
    print(f"  Fairness:        {scores['fairness']['score']:>3}/40  "
          f"{scores['fairness']['status']}")
    print(f"  Compliance:      {scores['compliance']['score']:>3}/30  "
          f"{scores['compliance']['status']}")
    print(f"  {'─'*50}")
    print(f"  TOTAL SCORE:     {scores['total']['score']:>3}/100")
    print(f"  GRADE:           {scores['total']['grade']}")
    print(f"  STATUS:          {scores['total']['status']}")
    print(f"{'='*60}")


# ── Test it ──
if __name__ == "__main__":
    print("Testing Responsible AI Score Module...")
    scores = compute_responsible_ai_score(
        bias_score=10,
        compliance_score=82,
        shap_works=True,
        lime_works=True,
        proxy_risks=[
            {"feature": "age"},
            {"feature": "duration"},
            {"feature": "dependents"}
        ],
        gdpr_results=[]
    )
    print_score_card(scores)