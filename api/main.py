# FastAPI Backend - Complete Fixed Version
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import json
import os
import sys
sys.path.append(".")

from agents.interpretability_agent import InterpretabilityAgent
from agents.lime_agent import LimeAgent
from agents.bias_detector_agent import BiasDetectorAgent
from orchestrator.orchestrator_agent import OrchestratorAgent
from orchestrator.responsible_ai_score import (
    compute_responsible_ai_score,
    save_audit_history
)

app = FastAPI(
    title="XAI Agent API",
    description="Rithvik's Explainable AI Agent!",
    version="1.0.0"
)

# ── 1. Load data FIRST ──
print("Loading data...")
X_test = pd.read_csv("data/X_test.csv")
X_train = pd.read_csv("data/X_train.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()
print("Data loaded ✅")

# ── 2. Load individual agents ──
print("Loading agents...")
shap_agent = InterpretabilityAgent()
shap_agent.load_model("data/credit_model.pkl", "data/feature_names.pkl")

lime_agent = LimeAgent()
lime_agent.load_model("data/credit_model.pkl", "data/feature_names.pkl", X_train)

bias_agent = BiasDetectorAgent()
bias_agent.load_model("data/credit_model.pkl", "data/feature_names.pkl")

# ── 3. Load orchestrator AFTER data ──
orchestrator = OrchestratorAgent()
orchestrator.load_all_agents(
    model_path="data/credit_model.pkl",
    feature_names_path="data/feature_names.pkl",
    X_train=X_train,
    rules_path="orchestrator/gdpr_rules.json"
)

# ── 4. Pre-compute bias at startup ──
print("Pre-computing bias results...")
sex_cols = [
    col for col in X_test.columns
    if 'personal_status_sex' in col
]
bias_cache = bias_agent.run(X_test, y_test, sex_cols)
print("All agents ready! ✅")

# ── Request model ──
class ExplainRequest(BaseModel):
    row_index: int = 0
    method: str = "both"

# ── Endpoints ──
@app.get("/")
def home():
    return {
        "message": "Welcome to Rithvik's XAI Agent API! 🚀",
        "endpoints": {
            "/explain": "Explain a prediction with SHAP and LIME",
            "/predict/{row_index}": "Quick prediction",
            "/bias": "Fairness audit results",
            "/score": "Get Responsible AI Score instantly",
            "/audit": "Run full audit with all 4 agents",
            "/audit/history": "View all past audits",
            "/health": "API health check"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "running ✅",
        "agents": ["SHAP", "LIME", "BiasDetector", "Compliance", "Orchestrator"],
        "test_samples": len(X_test),
        "fairness_score": bias_cache["fairness_score"]
    }

@app.post("/explain")
def explain(request: ExplainRequest):
    if request.row_index >= len(X_test) or request.row_index < 0:
        raise HTTPException(
            status_code=400,
            detail=f"row_index must be between 0 and {len(X_test)-1}"
        )
    row = X_test.iloc[[request.row_index]]
    result = {}

    if request.method in ["shap", "both"]:
        shap_result = shap_agent.explain_prediction(row, request.row_index)
        result["shap"] = {
            "prediction": shap_result["prediction"],
            "label": shap_result["label"],
            "confidence": shap_result["confidence"],
            "top_5_reasons": [
                {"feature": f, "impact": round(float(v), 4)}
                for f, v in shap_result["top_reasons"][:5]
            ]
        }

    if request.method in ["lime", "both"]:
        lime_result = lime_agent.explain_prediction(row, request.row_index)
        result["lime"] = {
            "prediction": lime_result["prediction"],
            "label": lime_result["label"],
            "confidence": lime_result["confidence"],
            "top_5_reasons": [
                {"feature": f, "impact": round(float(v), 4)}
                for f, v in lime_result["top_reasons"][:5]
            ]
        }

    if request.method == "both":
        agree = result["shap"]["prediction"] == result["lime"]["prediction"]
        result["agents_agree"] = agree
        result["verdict"] = "HIGH CONFIDENCE ✅" if agree else "REVIEW NEEDED ⚠️"

    result["row_index"] = request.row_index
    return result

@app.get("/predict/{row_index}")
def predict(row_index: int):
    if row_index >= len(X_test) or row_index < 0:
        raise HTTPException(
            status_code=400,
            detail=f"row_index must be between 0 and {len(X_test)-1}"
        )
    row = X_test.iloc[[row_index]]
    prediction = shap_agent.model.predict(row)[0]
    probability = shap_agent.model.predict_proba(row)[0]
    label = "BAD CREDIT RISK" if prediction == 1 else "GOOD CREDIT RISK"
    confidence = probability[prediction] * 100
    return {
        "row_index": row_index,
        "prediction": int(prediction),
        "label": label,
        "confidence": round(confidence, 2)
    }

@app.get("/bias")
def check_bias():
    return {
        "fairness_score": int(bias_cache["fairness_score"]),
        "rating": (
            "FAIR" if bias_cache["fairness_score"] >= 80
            else "NEEDS IMPROVEMENT" if bias_cache["fairness_score"] >= 60
            else "BIASED"
        ),
        "summary": [
            {
                "attribute": r["attribute"],
                "demographic_parity_difference": r["demographic_parity_difference"],
                "dpd_severity": r["dpd_severity"],
                "equalized_odds_difference": r["equalized_odds_difference"],
                "eod_severity": r["eod_severity"]
            }
            for r in bias_cache["bias_results"]
        ],
        "proxy_risks": bias_cache["proxy_risks"]
    }

@app.get("/score")
def get_score():
    scores = compute_responsible_ai_score(
        bias_score=int(bias_cache["fairness_score"]),
        compliance_score=82,
        shap_works=True,
        lime_works=True,
        proxy_risks=bias_cache["proxy_risks"],
        gdpr_results=[]
    )
    return {
        "responsible_ai_score": scores["total"]["score"],
        "grade": scores["total"]["grade"],
        "status": scores["total"]["status"],
        "breakdown": {
            "explainability": f"{scores['explainability']['score']}/30",
            "fairness": f"{scores['fairness']['score']}/40",
            "compliance": f"{scores['compliance']['score']}/30"
        }
    }

@app.post("/audit")
def run_full_audit():
    report = orchestrator.run(
        task="Full Responsible AI Audit",
        X_test=X_test,
        y_test=y_test,
        domain="credit",
        num_samples=3
    )
    scores = compute_responsible_ai_score(
        bias_score=int(
            report["bias_results"]["fairness_score"]
        ),
        compliance_score=int(
            report["compliance_results"]["compliance_score"]
        ),
        shap_works=True,
        lime_works=True,
        proxy_risks=report["bias_results"]["proxy_risks"],
        gdpr_results=[]
    )
    save_audit_history({
        **report,
        "scores": scores
    })
    return {
        "status": "Audit complete ✅",
        "responsible_ai_score": scores["total"]["score"],
        "grade": scores["total"]["grade"],
        "breakdown": {
            "explainability": f"{scores['explainability']['score']}/30",
            "fairness": f"{scores['fairness']['score']}/40",
            "compliance": f"{scores['compliance']['score']}/30"
        },
        "agents_run": report["agents_run"],
        "top_remediation": [
            step["action"] for step in
            report["compliance_results"]["remediation_steps"][:3]
        ]
    }

@app.get("/audit/history")
def get_audit_history():
    history_path = "data/audit_history.json"
    if not os.path.exists(history_path):
        return {"message": "No audits run yet", "history": []}
    with open(history_path, "r") as f:
        history = json.load(f)
    return {
        "total_audits": len(history),
        "history": history
    }