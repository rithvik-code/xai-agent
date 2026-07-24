# Let me create a completely clean app.py with NO triple-quote issues
# I'll use proper string escaping

app_py_clean = r # XAI Agent — Responsible AI Audit Platform
# Production-ready Streamlit app with Chatbot, CSV Scanner, and Document Analyzer

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

# Imports
try:
    from orchestrator.orchestrator_agent import OrchestratorAgent
    from agents.report_agent import ReportAgent
    from agents.explanation_agent import ExplanationAgent
except ImportError as e:
    st.error(f"Backend import failed: {e}")
    st.stop()

# Page Config
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

# Session State
def init_session():
    defaults = {
        "audit_results": None,
        "chat_history": [],
        "csv_bias_results": None,
        "csv_summary": None,
        "explanation_text": None,
        "doc_text": None,
        "doc_name": None,
        "doc_analysis": None,
        "doc_analysis_type": None,
        "doc_context": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# Data Loaders
@st.cache_data
def load_test_data():
    return pd.read_csv("data/X_test.csv"), pd.read_csv("data/y_test.csv").squeeze()

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

# LLM Caller
@st.cache_resource
def get_llm_client():
    hf_token = os.getenv("HF_TOKEN")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if groq_key:
        from groq import Groq
        return {"type": "groq", "client": Groq(api_key=groq_key)}
    elif hf_token:
        from openai import OpenAI
        return {
            "type": "hf",
            "client": OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
        }
    return None


def call_llm(messages, max_tokens=500):
    client_info = get_llm_client()
    if not client_info:
        raise ValueError("No API key found — add HF_TOKEN or GROQ_API_KEY to Secrets")
    
    if client_info["type"] == "groq":
        response = client_info["client"].chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=max_tokens,
        )
    else:
        response = client_info["client"].chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
    return response.choices[0].message.content.strip()


# Chat Context Builder
def build_chat_context():
    ctx = (
        "You are an expert Responsible AI advisor embedded in the XAI Agent platform. "
        "You help users understand AI bias, GDPR compliance, EU AI Act requirements, "
        "SHAP and LIME explanations, and how to make AI systems more ethical and fair. "
        "Be conversational, direct, and explain things in plain English without jargon. "
        "If a user asks what to do next, give them a concrete actionable recommendation. "
        "If a user is confused, reassure them and break things down step by step."
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
            f"Domain: {domain}. "
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
    
    if st.session_state.doc_context:
        ctx += f"\n\nDOCUMENT CONTEXT: {st.session_state.doc_context}"
    
    return ctx


# ============================================================
# SECTION 1: MAIN AUDIT
# ============================================================

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    run_audit = st.button(
        "🚀 RUN FULL AUDIT", use_container_width=True, type="primary"
    )

if run_audit:
    with st.spinner("Running Responsible AI Audit... This may take a minute..."):
        try:
            X_test, y_test = load_test_data()
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
        except Exception as e:
            st.error(f"Audit failed: {e}")

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
        if st.session_state.explanation_text:
            st.markdown("---")
            st.markdown(st.session_state.explanation_text)

    with tab4:
        st.subheader("Download Full PDF Report")
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                try:
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
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
else:
    st.info("👆 Click 'RUN FULL AUDIT' to begin the analysis")


# ============================================================
# SECTION 2: CSV UPLOAD + CHATBOT
# ============================================================
st.markdown("---")
st.header("🔬 Upload Your Own Data + AI Assistant")

left_col, right_col = st.columns([1, 1])

# CSV Upload
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
                        st.success("✅ Bias check complete!")

                    except Exception as e:
                        st.error(f"Bias check failed: {e}")

        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.csv_bias_results:
        st.markdown("### 📊 Your Bias Results")
        results_df = pd.DataFrame(st.session_state.csv_bias_results)
        st.dataframe(results_df, use_container_width=True)
        st.info("💡 Ask the chatbot on the right to explain what these numbers mean!")


# Chatbot
with right_col:
    st.subheader("💬 AI Assistant")
    st.caption("Ask anything — I know your audit results and can explain everything")

    # Display chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    '<div class="chat-user">👤 <b>You:</b> ' + msg["content"] + '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="chat-ai">🤖 <b>AI:</b> ' + msg["content"] + '</div>',
                    unsafe_allow_html=True
                )
    else:
        welcome_msg = (
            "Hi! I'm your Responsible AI advisor. "
            "Run the audit above, upload your CSV, or just ask me anything about "
            "AI bias, GDPR compliance, or the EU AI Act. What would you like to know?"
        )
        st.markdown(
            '<div class="chat-ai">🤖 <b>AI:</b> ' + welcome_msg + '</div>',
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


# ============================================================
# SECTION 3: DOCUMENT ANALYZER
# ============================================================
st.markdown("---")
st.header("📄 Document Analyzer")
st.caption("Upload any PDF or Word document — the AI will analyze it for bias, compliance issues, and legal red flags")

doc_col1, doc_col2 = st.columns([1, 1])

with doc_col1:
    st.subheader("Upload Document")
    doc_file = st.file_uploader(
        "Upload PDF or Word document",
        type=["pdf", "docx", "txt"],
        key="doc_uploader"
    )

    if doc_file:
        doc_text = ""
        try:
            if doc_file.name.endswith(".pdf"):
                import PyPDF2
                reader = PyPDF2.PdfReader(doc_file)
                for page in reader.pages:
                    doc_text += page.extract_text() + "\n"

            elif doc_file.name.endswith(".docx"):
                import docx
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(doc_file.read())
                    tmp_path = tmp.name
                doc_obj = docx.Document(tmp_path)
                for para in doc_obj.paragraphs:
                    doc_text += para.text + "\n"
                os.unlink(tmp_path)

            elif doc_file.name.endswith(".txt"):
                doc_text = doc_file.read().decode("utf-8", errors="ignore")

            if doc_text.strip():
                st.success(f"✅ Read {len(doc_text)} characters from {doc_file.name}")
                st.session_state.doc_text = doc_text
                st.session_state.doc_name = doc_file.name

                with st.expander("Preview document text"):
                    st.text(doc_text[:1000] + "..." if len(doc_text) > 1000 else doc_text)
            else:
                st.error("Could not extract text from this document.")

        except Exception as e:
            st.error(f"Failed to read document: {e}")

    if st.session_state.doc_text:
        st.markdown("---")

        analysis_type = st.selectbox(
            "What would you like to analyze?",
            [
                "Full Compliance & Bias Analysis",
                "GDPR Compliance Check",
                "EU AI Act Risk Assessment",
                "Bias & Fairness Language Review",
                "Legal Red Flags",
                "Executive Summary"
            ]
        )

        if st.button("🔍 Analyze Document", type="primary"):
            with st.spinner("AI is reading and analyzing your document..."):
                try:
                    text_for_analysis = st.session_state.doc_text[:8000]
                    if len(st.session_state.doc_text) > 8000:
                        text_for_analysis += "\n[Document truncated for analysis]"

                    prompts = {
                        "Full Compliance & Bias Analysis": "Analyze this document for ALL of the following and provide a detailed structured report:\n\n1. GDPR COMPLIANCE: Check for mentions of data processing, consent, right to explanation, data minimization. Flag any violations or missing requirements.\n2. EU AI ACT RISK: Identify what type of AI system this describes and classify its risk tier (Unacceptable/High/Limited/Minimal).\n3. BIAS & FAIRNESS: Look for any language that could indicate demographic bias, unfair treatment, or discrimination against protected groups.\n4. LEGAL RED FLAGS: Identify any statements that could create legal liability.\n5. RECOMMENDATIONS: Give 3-5 specific actions to improve compliance.\n\nDocument: " + text_for_analysis + "\n\nProvide your analysis in clear sections with specific quotes from the document where relevant.",

                        "GDPR Compliance Check": "You are a GDPR compliance expert. Analyze this document and check:\n- Article 5: Data minimization and purpose limitation\n- Article 13/14: Transparency and information obligations  \n- Article 22: Automated decision-making rights\n- Article 25: Privacy by design\n- Article 35: Data Protection Impact Assessment requirements\n\nFor each article, state: COMPLIANT / AT RISK / NON-COMPLIANT with specific reasoning.\n\nDocument: " + text_for_analysis,

                        "EU AI Act Risk Assessment": "You are an EU AI Act expert. Analyze this document and:\n1. Identify what AI system or use case is described\n2. Classify the risk tier: Unacceptable / High / Limited / Minimal\n3. List the specific obligations that apply\n4. Identify any compliance gaps\n5. Give a deployment recommendation\n\nDocument: " + text_for_analysis,

                        "Bias & Fairness Language Review": "You are a fairness and bias expert. Analyze this document for:\n1. Any language that discriminates against protected groups (gender, race, age, religion, disability)\n2. Proxy variables that could indirectly encode bias\n3. Missing fairness considerations\n4. Recommendations to make the document more equitable\n\nBe specific — quote exact phrases that are problematic.\n\nDocument: " + text_for_analysis,

                        "Legal Red Flags": "You are a legal risk analyst. Review this document and identify:\n1. Statements that could create legal liability\n2. Missing required disclosures\n3. Contradictions or ambiguities that could be exploited\n4. Regulatory violations\n5. Risk severity for each finding (HIGH/MEDIUM/LOW)\n\nDocument: " + text_for_analysis,

                        "Executive Summary": "Create a professional executive summary of this document covering:\n1. What this document is about (2-3 sentences)\n2. Key AI/data practices described\n3. Main compliance strengths\n4. Main compliance risks\n5. Overall risk rating (HIGH/MEDIUM/LOW) with justification\n\nKeep it concise and suitable for a non-technical executive audience.\n\nDocument: " + text_for_analysis
                    }

                    messages = [
                        {
                            "role": "system",
                            "content": "You are an expert AI compliance analyst specializing in GDPR, EU AI Act, fairness, and responsible AI. Provide thorough, honest, actionable analysis."
                        },
                        {
                            "role": "user",
                            "content": prompts[analysis_type]
                        }
                    ]

                    analysis_result = call_llm(messages, max_tokens=1500)
                    st.session_state.doc_analysis = analysis_result
                    st.session_state.doc_analysis_type = analysis_type
                    st.session_state.doc_context = f"The user uploaded '{st.session_state.doc_name}' and got a {analysis_type}. Result: {analysis_result[:500]}..."
                    st.success("✅ Analysis complete!")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

with doc_col2:
    st.subheader("Analysis Results")

    if st.session_state.doc_analysis:
        st.markdown(f"**{st.session_state.doc_analysis_type}**")
        st.markdown("---")
        st.markdown(st.session_state.doc_analysis)

        st.markdown("---")
        st.info("💬 Ask the AI Assistant above any follow-up questions about this document!")

        st.download_button(
            "⬇️ Download Analysis",
            st.session_state.doc_analysis,
            file_name=f"analysis_{st.session_state.get('doc_name','document')}.txt",
            mime="text/plain"
        )
    else:
        placeholder_msg = (
            "Upload a document on the left and click "
            "Analyze Document to get a detailed compliance and bias analysis."
        )
        st.markdown(
            '<div class="chat-ai">🤖 ' + placeholder_msg + '</div>',
            unsafe_allow_html=True
        )


