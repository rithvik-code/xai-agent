# Phase 5 - Streamlit Frontend with Chatbot + CSV Upload
import streamlit as st
import pandas as pd
import sys
import os
import json
import tempfile
from dotenv import load_dotenv

# Lock working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.append(BASE_DIR)
load_dotenv()

from orchestrator.orchestrator_agent import OrchestratorAgent
from agents.report_agent import ReportAgent
from agents.explanation_agent import ExplanationAgent

# Page config
st.set_page_config(
    page_title="XAI Agent - Responsible AI Audit",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; }
    h1, h2, h3 { color: #e0e0e0; }
    .chat-user {
        background-color: #1a1a2e;
        border-left: 4px solid #7c5cbf;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e0e0e0;
    }
    .chat-ai {
        background-color: #0d1117;
        border-left: 4px solid #28a745;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e0e0e0;
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
    st.markdown("---")
    st.markdown("**Quick Help:**")
    st.markdown("1. Run the audit on the built-in credit model")
    st.markdown("2. Upload your own CSV for bias checking")
    st.markdown("3. Ask the chatbot anything about your results")

# Initialize session state
if "audit_results" not in st.session_state:
    st.session_state.audit_results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "csv_bias_results" not in st.session_state:
    st.session_state.csv_bias_results = None
if "csv_summary" not in st.session_state:
    st.session_state.csv_summary = None


def load_data():
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    return X_test, y_test


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


def build_chat_context():
    ctx = (
        "You are an expert Responsible AI advisor embedded in the XAI Agent platform. "
        "You help users understand AI bias, GDPR compliance, EU AI Act requirements, "
        "SHAP and LIME explanations, and how to make AI systems more ethical and fair. "
        "Be conversational, direct, and explain things in plain English without jargon. "
        "If a user asks what to do next, give them a concrete actionable recommendation. "
        "If a user is confused, reassure them and break things down step by step. "
    )
    if st.session_state.audit_results:
        scores = st.session_state.audit_results["scores"]
        results = st.session_state.audit_results["results"]
        ctx += (
            f"\n\nCURRENT AUDIT CONTEXT: The user just ran a full Responsible AI audit. "
            f"Overall Score: {scores['total']['score']}/100, "
            f"Grade: {scores['total']['grade']}, "
            f"Status: {scores['total']['status']}. "
            f"Explainability: {scores['explainability']['score']}/30. "
            f"Fairness: {scores['fairness']['score']}/40. "
            f"Compliance: {scores['compliance']['score']}/30. "
            f"Domain: credit. "
            f"EU AI Act Risk Tier: {results['compliance_results']['risk_tier']['tier'].upper()}. "
        )
        bias_summary = ", ".join([
            f"{r['attribute']}: DPD={r['demographic_parity_difference']} ({r['dpd_severity']})"
            for r in results["bias_results"]["bias_results"]
        ])
        ctx += f"Bias findings: {bias_summary}. "

    if st.session_state.csv_summary:
        s = st.session_state.csv_summary
        ctx += (
            f"\n\nUSER CSV CONTEXT: The user uploaded a dataset with {s['rows']} rows "
            f"and {s['cols']} columns. Target column: '{s['target']}'. "
            f"Protected columns checked: {', '.join(s['protected'])}. "
            f"Bias results: {json.dumps(s['results'])}. "
        )
    return ctx


def call_llm(messages):
    hf_token = os.getenv("HF_TOKEN")
    groq_key = os.getenv("GROQ_API_KEY")

    if hf_token:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token
        )
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
    elif groq_key:
        from groq import Groq
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500,
        )
    else:
        raise ValueError("No API key found — add HF_TOKEN or GROQ_API_KEY to your .env file")

    return response.choices[0].message.content.strip()


# ==================== MAIN AUDIT SECTION ====================
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
        st.session_state.audit_results = {"results": results, "scores": scores}
    st.success("✅ Audit Complete!")

if st.session_state.audit_results:
    results = st.session_state.audit_results["results"]
    scores = st.session_state.audit_results["scores"]

    st.markdown("---")
    st.header("📊 Audit Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Responsible AI Score", f"{scores['total']['score']}/100", scores['total']['grade'])
    with col2:
        st.metric("Explainability", f"{scores['explainability']['score']}/30")
    with col3:
        st.metric("Fairness", f"{scores['fairness']['score']}/40")
    with col4:
        st.metric("Compliance", f"{scores['compliance']['score']}/30")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Bias Analysis", "📜 Compliance",
        "📝 Plain-English Summary", "📄 Download Report"
    ])

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
        st.subheader("Plain-English Summary")
        st.caption("Powered by Llama via HuggingFace / Groq")
        if st.button("🧠 Generate Explanation"):
            with st.spinner("Translating technical results into plain English..."):
                try:
                    explainer = ExplanationAgent()
                    explanation = explainer.explain(
                        scores=scores,
                        bias_results=results["bias_results"],
                        compliance_results=results["compliance_results"],
                        domain=domain
                    )
                    st.session_state.explanation_text = explanation
                except Exception as e:
                    st.error(f"Couldn't generate explanation: {e}")
        if "explanation_text" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state.explanation_text)

    with tab4:
        st.subheader("Download Full PDF Report")
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                report_agent = ReportAgent()
                pdf_path = report_agent.generate_pdf(
                    scores=scores,
                    bias_results=results["bias_results"],
                    compliance_results=results["compliance_results"],
                    model_accuracy=79.5,
                    domain=domain,
                    explanation=st.session_state.get("explanation_text", None)
                )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF Report", f,
                        file_name="XAI_Audit_Report.pdf",
                        mime="application/pdf"
                    )
            st.success("✅ Report generated!")
else:
    st.info("👆 Click 'RUN FULL AUDIT' to begin the analysis")

# ==================== CSV UPLOAD + CHATBOT SECTION ====================
st.markdown("---")
st.header("🔬 Upload Your Own Data + AI Assistant")

left_col, right_col = st.columns([1, 1])

# ---- CSV UPLOAD ----
with left_col:
    st.subheader("📁 Bias Check on Your CSV")
    st.caption("Upload any dataset to check it for demographic bias")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded: {len(user_df)} rows × {len(user_df.columns)} columns")
            st.dataframe(user_df.head(5), use_container_width=True)

            target_col = st.selectbox(
                "Target column (what the model predicts):",
                user_df.columns.tolist()
            )
            protected_cols = st.multiselect(
                "Protected attribute columns (gender, age, race, etc):",
                [c for c in user_df.columns if c != target_col]
            )

            if st.button("🔍 Run Bias Check", type="primary") and protected_cols:
                with st.spinner("Training a quick model and checking for bias..."):
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        from fairlearn.metrics import (
                            demographic_parity_difference,
                            equalized_odds_difference
                        )

                        feature_cols = [c for c in user_df.columns if c != target_col]
                        X = pd.get_dummies(user_df[feature_cols].fillna(0))
                        y = user_df[target_col]

                        clf = RandomForestClassifier(n_estimators=100, random_state=42)
                        clf.fit(X, y)
                        preds = clf.predict(X)

                        bias_results_csv = []
                        for col in protected_cols:
                            try:
                                sensitive = user_df[col]
                                dpd = round(demographic_parity_difference(
                                    y, preds, sensitive_features=sensitive
                                ), 4)
                                eod = round(equalized_odds_difference(
                                    y, preds, sensitive_features=sensitive
                                ), 4)
                                status_label = (
                                    "✅ PASS" if abs(dpd) < 0.1
                                    else "⚠️ WARN" if abs(dpd) < 0.2
                                    else "❌ FAIL"
                                )
                                bias_results_csv.append({
                                    "Column": col,
                                    "DPD": dpd,
                                    "EOD": eod,
                                    "Status": status_label
                                })
                            except Exception:
                                bias_results_csv.append({
                                    "Column": col,
                                    "DPD": "N/A",
                                    "EOD": "N/A",
                                    "Status": "⚠️ Could not compute"
                                })

                        st.session_state.csv_bias_results = bias_results_csv
                        st.session_state.csv_summary = {
                            "rows": len(user_df),
                            "cols": len(user_df.columns),
                            "target": target_col,
                            "protected": protected_cols,
                            "results": bias_results_csv
                        }

                    except Exception as e:
                        st.error(f"Bias check failed: {e}")

        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.csv_bias_results:
        st.markdown("### 📊 Your Bias Results")
        results_df = pd.DataFrame(st.session_state.csv_bias_results)
        st.dataframe(results_df, use_container_width=True)
        st.info("💡 Ask the chatbot on the right to explain what these numbers mean!")

# ---- CHATBOT ----
with right_col:
    st.subheader("💬 AI Assistant")
    st.caption("Ask anything — I know your audit results and can explain everything")

    # Display chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">👤 <b>You:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-ai">🤖 <b>AI:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.markdown(
            '<div class="chat-ai">🤖 <b>AI:</b> Hi! I\'m your Responsible AI advisor. '
            'Run the audit above, upload your CSV, or just ask me anything about '
            'AI bias, GDPR compliance, or the EU AI Act. What would you like to know?</div>',
            unsafe_allow_html=True
        )

    # Suggested questions
    st.markdown("**Quick questions:**")
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        if st.button("What does my score mean?", use_container_width=True):
            st.session_state.pending_question = "What does my Responsible AI Score mean and is my model safe to deploy?"
    with q_col2:
        if st.button("Explain the bias findings", use_container_width=True):
            st.session_state.pending_question = "Can you explain the bias findings in simple terms and what I should do about them?"

    q_col3, q_col4 = st.columns(2)
    with q_col3:
        if st.button("What is GDPR Article 22?", use_container_width=True):
            st.session_state.pending_question = "What is GDPR Article 22 and how does it apply to my model?"
    with q_col4:
        if st.button("What is the EU AI Act?", use_container_width=True):
            st.session_state.pending_question = "Can you explain the EU AI Act and what HIGH RISK means for my system?"

    # Text input
    user_input = st.text_input(
        "Type your question:",
        placeholder="e.g. Why did my model fail the fairness check?",
        key="chat_input_box"
    )

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        send_btn = st.button("Send 💬", use_container_width=True, type="primary")
    with btn_col2:
        if st.button("Clear 🗑️", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Handle pending question from quick buttons
    question_to_send = None
    if send_btn and user_input:
        question_to_send = user_input
    elif "pending_question" in st.session_state:
        question_to_send = st.session_state.pending_question
        del st.session_state.pending_question

    if question_to_send:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question_to_send
        })

        with st.spinner("Thinking..."):
            try:
                context = build_chat_context()
                messages = [{"role": "system", "content": context}]
                # Include last 6 messages for context
                for msg in st.session_state.chat_history[-6:]:
                    messages.append({
                        "role": "user" if msg["role"] == "user" else "assistant",
                        "content": msg["content"]
                    })
                reply = call_llm(messages)
            except Exception as e:
                reply = (
                    f"I couldn't connect to the AI right now ({str(e)[:80]}). "
                    "However, I can still help — your audit results are loaded and "
                    "I have context about your findings. Try again in a moment!"
                )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply
        })
        st.rerun()
