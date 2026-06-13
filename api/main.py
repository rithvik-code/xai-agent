# Step 6 - FastAPI Backend
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
import os

# So Python can find our agents folder
sys.path.append(".")
from agents.interpretability_agent import InterpretabilityAgent
from agents.lime_agent import LimeAgent

# ── Create the app ──
app = FastAPI(
    title="XAI Agent API",
    description="Rithvik's Explainable AI Agent — explains AI decisions!",
    version="1.0.0"
)

# ── Load everything once when server starts ──
print("Loading models and agents...")
X_test = pd.read_csv("data/X_test.csv")
X_train = pd.read_csv("data/X_train.csv")

shap_agent = InterpretabilityAgent()
shap_agent.load_model("data/credit_model.pkl", "data/feature_names.pkl")

lime_agent = LimeAgent()
lime_agent.load_model("data/credit_model.pkl", "data/feature_names.pkl", X_train)

print("All agents ready! ✅")

# ── Request model ──
class ExplainRequest(BaseModel):
    row_index: int = 0
    method: str = "both"  # "shap", "lime", or "both"

# ── Endpoints ──
@app.get("/")
def home():
    return {
        "message": "Welcome to Rithvik's XAI Agent API! 🚀",
        "endpoints": {
            "/explain": "Explain a prediction",
            "/predict": "Get a prediction only",
            "/health": "Check if API is running"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "running ✅",
        "agents": ["SHAP", "LIME"],
        "test_samples": len(X_test)
    }

@app.post("/explain")
def explain(request: ExplainRequest):
    # Check valid row index
    if request.row_index >= len(X_test) or request.row_index < 0:
        raise HTTPException(
            status_code=400,
            detail=f"row_index must be between 0 and {len(X_test)-1}"
        )

    row = X_test.iloc[[request.row_index]]
    result = {}

    # Run SHAP
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

    # Run LIME
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

    # Add agreement check if both ran
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