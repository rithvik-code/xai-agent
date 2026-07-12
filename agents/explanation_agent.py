# Phase 5 Step 2 - NL Explanation Agent (HuggingFace Inference API)

import os

from dotenv import load_dotenv

from huggingface_hub import InferenceClient



load_dotenv()





class ExplanationAgent:

    def __init__(self):

        api_key = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY")

        if not api_key:

            raise ValueError(

                "HF_TOKEN not found! Add it to your .env file or "

                "Space secrets."

            )

        # Try HF token first, fall back to Groq if available

        hf_token = os.getenv("HF_TOKEN")

        if hf_token:

            self.client = InferenceClient(

                model="meta-llama/Meta-Llama-3-8B-Instruct",  # ✅ supported model

                token=hf_token

            )

            self.backend = "huggingface"

        else:

            from groq import Groq

            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            self.backend = "groq"

        print(f"ExplanationAgent initialized via {self.backend} ✅")



    def explain(self, scores, bias_results,

                compliance_results, domain="credit"):

        """Generate plain-English explanation of audit results"""



        total_score = scores["total"]["score"]

        grade = scores["total"]["grade"]

        fairness_score = bias_results["fairness_score"]

        compliance_score = compliance_results["compliance_score"]

        risk_tier = compliance_results["risk_tier"]["tier"]



        bias_summary = "\n".join([

            f"- {r['attribute']}: DPD={r['demographic_parity_difference']}"

            f" ({r['dpd_severity']}), "

            f"EOD={r['equalized_odds_difference']} ({r['eod_severity']})"

            for r in bias_results["bias_results"]

        ])



        proxy_summary = "\n".join([

            f"- {p['feature']} correlates with "

            f"{p['protected_attribute']} "

            f"(correlation={p['correlation']}, {p['risk']})"

            for p in bias_results.get("proxy_risks", [])

        ]) or "None detected"



        prompt = f"""You are a responsible AI explainer.

Translate this technical XAI audit output into plain English for a

non-technical audience (e.g. a business executive or compliance officer).



Domain: {domain}

EU AI Act Risk Tier: {risk_tier}

Responsible AI Score: {total_score}/100 (Grade: {grade})

Fairness Score: {fairness_score}/100

Compliance Score: {compliance_score}/100



Bias findings:

{bias_summary}



Proxy features detected:

{proxy_summary}



Write exactly 2 paragraphs in plain English.



Paragraph 1: Explain what the overall score means and whether this

model is safe to deploy right now. Be direct.



Paragraph 2: Explain the bias findings in simple terms a non-technical

person can understand, and give 2-3 specific actions to fix them.



No bullet points. No technical jargon. Write like you are explaining

to a smart business executive who has never heard of SHAP or fairness

metrics."""



        if self.backend == "huggingface":

            response = self.client.chat_completion(

                messages=[{"role": "user", "content": prompt}],

                max_tokens=500,

                temperature=0.7,

            )

            explanation = response.choices[0].message["content"].strip()

        else:

            response = self.client.chat.completions.create(

                model="llama-3.3-70b-versatile",

                max_tokens=500,

                messages=[{"role": "user", "content": prompt}]

            )

            explanation = response.choices[0].message.content



        print("\n📝 PLAIN-ENGLISH EXPLANATION:")

        print("=" * 50)

        print(explanation)

        print("=" * 50)



        return explanation