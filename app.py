# ==================== DOCUMENT ANALYZER SECTION ====================
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
        # Extract text based on file type
        doc_text = ""
        try:
            if doc_file.name.endswith(".pdf"):
                import PyPDF2
                reader = PyPDF2.PdfReader(doc_file)
                for page in reader.pages:
                    doc_text += page.extract_text() + "\n"

            elif doc_file.name.endswith(".docx"):
                import docx
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(doc_file.read())
                    tmp_path = tmp.name
                doc_obj = docx.Document(tmp_path)
                for para in doc_obj.paragraphs:
                    doc_text += para.text + "\n"

            elif doc_file.name.endswith(".txt"):
                doc_text = doc_file.read().decode("utf-8", errors="ignore")

            if doc_text.strip():
                st.success(f"✅ Read {len(doc_text)} characters from {doc_file.name}")
                st.session_state.doc_text = doc_text
                st.session_state.doc_name = doc_file.name

                # Show preview
                with st.expander("Preview document text"):
                    st.text(doc_text[:1000] + "..." if len(doc_text) > 1000 else doc_text)
            else:
                st.error("Could not extract text from this document.")

        except Exception as e:
            st.error(f"Failed to read document: {e}")

    if "doc_text" in st.session_state and st.session_state.doc_text:
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
                    # Truncate if too long
                    text_for_analysis = st.session_state.doc_text[:8000]
                    if len(st.session_state.doc_text) > 8000:
                        text_for_analysis += "\n[Document truncated for analysis]"

                    prompts = {
                        "Full Compliance & Bias Analysis": f"""Analyze this document for ALL of the following and provide a detailed structured report:

1. GDPR COMPLIANCE: Check for mentions of data processing, consent, right to explanation, data minimization. Flag any violations or missing requirements.
2. EU AI ACT RISK: Identify what type of AI system this describes and classify its risk tier (Unacceptable/High/Limited/Minimal).
3. BIAS & FAIRNESS: Look for any language that could indicate demographic bias, unfair treatment, or discrimination against protected groups.
4. LEGAL RED FLAGS: Identify any statements that could create legal liability.
5. RECOMMENDATIONS: Give 3-5 specific actions to improve compliance.

Document: {text_for_analysis}

Provide your analysis in clear sections with specific quotes from the document where relevant.""",

                        "GDPR Compliance Check": f"""You are a GDPR compliance expert. Analyze this document and check:
- Article 5: Data minimization and purpose limitation
- Article 13/14: Transparency and information obligations  
- Article 22: Automated decision-making rights
- Article 25: Privacy by design
- Article 35: Data Protection Impact Assessment requirements

For each article, state: COMPLIANT / AT RISK / NON-COMPLIANT with specific reasoning.

Document: {text_for_analysis}""",

                        "EU AI Act Risk Assessment": f"""You are an EU AI Act expert. Analyze this document and:
1. Identify what AI system or use case is described
2. Classify the risk tier: Unacceptable / High / Limited / Minimal
3. List the specific obligations that apply
4. Identify any compliance gaps
5. Give a deployment recommendation

Document: {text_for_analysis}""",

                        "Bias & Fairness Language Review": f"""You are a fairness and bias expert. Analyze this document for:
1. Any language that discriminates against protected groups (gender, race, age, religion, disability)
2. Proxy variables that could indirectly encode bias
3. Missing fairness considerations
4. Recommendations to make the document more equitable

Be specific — quote exact phrases that are problematic.

Document: {text_for_analysis}""",

                        "Legal Red Flags": f"""You are a legal risk analyst. Review this document and identify:
1. Statements that could create legal liability
2. Missing required disclosures
3. Contradictions or ambiguities that could be exploited
4. Regulatory violations
5. Risk severity for each finding (HIGH/MEDIUM/LOW)

Document: {text_for_analysis}""",

                        "Executive Summary": f"""Create a professional executive summary of this document covering:
1. What this document is about (2-3 sentences)
2. Key AI/data practices described
3. Main compliance strengths
4. Main compliance risks
5. Overall risk rating (HIGH/MEDIUM/LOW) with justification

Keep it concise and suitable for a non-technical executive audience.

Document: {text_for_analysis}"""
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

                    analysis_result = call_llm(messages)
                    st.session_state.doc_analysis = analysis_result
                    st.session_state.doc_analysis_type = analysis_type

                    # Add to chat context
                    st.session_state.doc_context = f"The user uploaded '{st.session_state.doc_name}' and got a {analysis_type}. Result: {analysis_result[:500]}..."

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

with doc_col2:
    st.subheader("Analysis Results")

    if "doc_analysis" in st.session_state:
        st.markdown(f"**{st.session_state.doc_analysis_type}**")
        st.markdown("---")
        st.markdown(st.session_state.doc_analysis)

        st.markdown("---")
        st.info("💬 Ask the AI Assistant above any follow-up questions about this document!")

        # Download analysis as text
        st.download_button(
            "⬇️ Download Analysis",
            st.session_state.doc_analysis,
            file_name=f"analysis_{st.session_state.get('doc_name','document')}.txt",
            mime="text/plain"
        )
    else:
        st.markdown(
            '<div class="chat-ai">🤖 Upload a document on the left and click '
            '"Analyze Document" to get a detailed compliance and bias analysis.</div>',
            unsafe_allow_html=True
        )
