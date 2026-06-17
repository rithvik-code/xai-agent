# Phase 5 - Streamlit Frontend
import streamlit as st
import pandas as pd
import sys
import os

# Lock working directory to this script's folder no matter where streamlit is launched from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)

from orchestrator.orchestrator_agent import OrchestratorAgent
from orchestrator.responsible_ai_score import compute_responsible_ai_score
from agents.report_agent import ReportAgent

# Page config
st.set_page_config(
    page_title="XAI Agent - Responsible AI Audit",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0f0f1a;
    }
    .stApp {
        background-color: #0f0f1a;
    }
    h1, h2, h3 {
        color: #e0e0e0;
    }
    .metric-card {
        background-color: #1a1a2e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #7c5cbf;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 XAI Agent — Responsible AI Audit Platform")
st.markdown("### Built by Rithvik | SHAP + LIME + Fairness + GDPR Compliance")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    domain = st.selectbox(
        "Select Domain",
        ["credit", "healthcare", "hiring", "law_enforcement"]
    )
    num_samples = st.slider("Number of samples to explain", 1, 10, 3)
    st.markdown("---")
    st.markdown("**About this tool:**")
    st.markdown(
        "This platform audits ML models for explainability, "
        "fairness, and regulatory compliance (GDPR + EU AI Act)."
    )

# Initialize session state
if "audit_results" not in st.session_state:
    st.session_state.audit_results = None

# Load data
@st.cache_resource
def load_data():
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    return X_test, y_test

@st.cache_resource
def load_orchestrator():
    X_train = pd.read_csv("data/X_train.csv")
    orch = OrchestratorAgent()
    orch.load_all_agents(
        model_path="data/credit_model.pkl",
        feature_names_path="data/feature_names.pkl",
        X_train=X_train,
        rules_path="orchestrator/gdpr_rules.json"
    )
    return orch

# Main button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    run_audit = st.button(
        "🚀 RUN FULL AUDIT", use_container_width=True, type="primary"
    )

if run_audit:
    with st.spinner("Running Responsible AI Audit... This may take a minute..."):
        X_test, y_test = load_data()
        orchestrator = load_orchestrator()

        results = orchestrator.run(
            task="credit_audit",
            X_test=X_test,
            y_test=y_test,
            domain=domain,
            num_samples=num_samples
        )

        # Use the score dict already computed inside orchestrator.run()
        raw_scores = results["scores"]
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

        st.session_state.audit_results = {
            "results": results,
            "scores": scores
        }

    st.success("✅ Audit Complete!")

# Display results
if st.session_state.audit_results:
    results = st.session_state.audit_results["results"]
    scores = st.session_state.audit_results["scores"]

    st.markdown("---")
    st.header("📊 Audit Results")

    # Score cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Responsible AI Score",
            f"{scores['total']['score']}/100",
            scores['total']['grade']
        )
    with col2:
        st.metric(
            "Explainability",
            f"{scores['explainability']['score']}/30"
        )
    with col3:
        st.metric(
            "Fairness",
            f"{scores['fairness']['score']}/40"
        )
    with col4:
        st.metric(
            "Compliance",
            f"{scores['compliance']['score']}/30"
        )

    st.markdown("---")

    # Tabs for details
    tab1, tab2, tab3 = st.tabs(
        ["🎯 Bias Analysis", "📜 Compliance", "📄 Download Report"]
    )

    with tab1:
        st.subheader("Fairness Metrics")
        bias_data = []
        for r in results["bias_results"]["bias_results"]:
            bias_data.append({
    "Attribute": r["attribute"],
    "DPD": r["demographic_parity_difference"],
    "DPD Status": r["dpd_severity"],
    "EOD": r["equalized_odds_difference"],
    "EOD Status": r["eod_severity"]
})
        st.dataframe(pd.DataFrame(bias_data), use_container_width=True)

        st.subheader("Proxy Features Detected")
        if results["bias_results"]["proxy_risks"]:
            proxy_data = []
            for p in results["bias_results"]["proxy_risks"]:
                proxy_data.append({
                    "Feature": p["feature"],
                    "Protected Attribute": p["protected_attribute"],
                    "Correlation": p["correlation"],
                    "Risk": p["risk"]
                })
            st.dataframe(pd.DataFrame(proxy_data), use_container_width=True)
        else:
            st.info("No proxy features detected.")

    with tab2:
        st.subheader("GDPR Compliance")
        gdpr_data = []
        for r in results["compliance_results"]["gdpr_results"]:
            gdpr_data.append({
                "Rule": r["rule_name"],
                "Status": r["status"],
                "Finding": r["finding"]
            })
        st.dataframe(pd.DataFrame(gdpr_data), use_container_width=True)

        risk_tier = results["compliance_results"]["risk_tier"]
        st.warning(
            f"**EU AI Act Risk Tier:** {risk_tier['tier'].upper()} — "
            f"{risk_tier.get('action', '')}"
        )

    with tab3:
        st.subheader("Download Full PDF Report")
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                report_agent = ReportAgent()
                pdf_path = report_agent.generate_pdf(
                    scores=scores,
                    bias_results=results["bias_results"],
                    compliance_results=results["compliance_results"],
                    model_accuracy=79.5,
                    domain=domain
                )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        f,
                        file_name="XAI_Audit_Report.pdf",
                        mime="application/pdf"
                    )
            st.success("✅ Report generated!")
else:
    st.info("👆 Click 'RUN FULL AUDIT' to begin the analysis")