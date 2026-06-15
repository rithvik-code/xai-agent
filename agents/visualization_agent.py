# Phase 4 - Visualization Agent
import pandas as pd
import numpy as np
import pickle
import os
import sys
sys.path.append(".")
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class VisualizationAgent:
    def __init__(self):
        self.output_dir = "reports"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/html", exist_ok=True)
        os.makedirs(f"{self.output_dir}/img", exist_ok=True)
        print("VisualizationAgent initialized ✅")

    def plot_shap_bar(self, shap_results, row_index=0):
        """Interactive SHAP bar chart using Plotly"""
        result = shap_results[row_index]
        features = [r[0] for r in result["top_reasons"][:10]]
        values = [float(r[1]) for r in result["top_reasons"][:10]]
        colors = [
            "#ff6b6b" if v > 0 else "#51cf66"
            for v in values
        ]
        fig = go.Figure(go.Bar(
            x=values[::-1],
            y=features[::-1],
            orientation='h',
            marker_color=colors[::-1],
            hovertemplate="<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>"
        ))
        fig.update_layout(
            title=f"SHAP Explanation — Row {row_index}<br>"
                  f"<sub>Prediction: {result['label']} "
                  f"({result['confidence']}% confidence)</sub>",
            xaxis_title="SHAP Value (Impact on prediction)",
            yaxis_title="Feature",
            height=500,
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(color="white")
        )
        # Add vertical line at 0
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        path = f"{self.output_dir}/html/shap_row{row_index}.html"
        fig.write_html(path)
        print(f"SHAP chart saved: {path} ✅")
        return path

    def plot_fairness_dashboard(self, bias_results):
        """Interactive fairness dashboard"""
        bias_data = bias_results["bias_results"]

        attributes = [r["attribute"] for r in bias_data]
        dpd_values = [r["demographic_parity_difference"] for r in bias_data]
        eod_values = [r["equalized_odds_difference"] for r in bias_data]
        dpd_colors = [
            "#51cf66" if abs(v) < 0.05
            else "#ffd43b" if abs(v) < 0.10
            else "#ff6b6b"
            for v in dpd_values
        ]
        eod_colors = [
            "#51cf66" if abs(v) < 0.05
            else "#ffd43b" if abs(v) < 0.10
            else "#ff6b6b"
            for v in eod_values
        ]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Demographic Parity Difference",
                "Equalized Odds Difference"
            ]
        )

        fig.add_trace(
            go.Bar(
                x=attributes,
                y=dpd_values,
                marker_color=dpd_colors,
                name="DPD",
                hovertemplate="<b>%{x}</b><br>DPD: %{y:.4f}<extra></extra>"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=attributes,
                y=eod_values,
                marker_color=eod_colors,
                name="EOD",
                hovertemplate="<b>%{x}</b><br>EOD: %{y:.4f}<extra></extra>"
            ),
            row=1, col=2
        )

        # Add threshold lines
        for col in [1, 2]:
            fig.add_hline(
                y=0.05, line_dash="dash",
                line_color="#51cf66",
                annotation_text="PASS threshold",
                row=1, col=col
            )
            fig.add_hline(
                y=0.10, line_dash="dash",
                line_color="#ffd43b",
                annotation_text="WARN threshold",
                row=1, col=col
            )

        fig.update_layout(
            title="Fairness Dashboard — Gender Bias Analysis",
            height=500,
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(color="white"),
            showlegend=False
        )

        path = f"{self.output_dir}/html/fairness_dashboard.html"
        fig.write_html(path)
        print(f"Fairness dashboard saved: {path} ✅")
        return path

    def plot_compliance_heatmap(self, compliance_results):
        """Compliance traffic light heat map"""
        gdpr = compliance_results["gdpr_results"]

        rules = [r["rule_name"] for r in gdpr]
        statuses = [r["status"] for r in gdpr]
        findings = [r["finding"] for r in gdpr]

        # Convert status to number for color
        status_num = []
        for s in statuses:
            if "COMPLIANT ✅" in s:
                status_num.append(1)
            elif "AT_RISK" in s or "PARTIAL" in s or "REQUIRED" in s:
                status_num.append(0.5)
            else:
                status_num.append(0)

        fig = go.Figure(go.Heatmap(
            z=[status_num],
            x=rules,
            y=["GDPR Status"],
            colorscale=[
                [0, "#ff6b6b"],
                [0.5, "#ffd43b"],
                [1, "#51cf66"]
            ],
            showscale=False,
            hovertemplate="<b>%{x}</b><br>Status: %{text}<extra></extra>",
            text=[statuses]
        ))

        # Add status text on cells
        for i, (rule, status, finding) in enumerate(
            zip(rules, statuses, findings)
        ):
            fig.add_annotation(
                x=rule, y="GDPR Status",
                text=status.split()[0],
                showarrow=False,
                font=dict(color="black", size=10)
            )

        fig.update_layout(
            title="GDPR Compliance Heat Map",
            height=250,
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            font=dict(color="white")
        )

        path = f"{self.output_dir}/html/compliance_heatmap.html"
        fig.write_html(path)
        print(f"Compliance heatmap saved: {path} ✅")
        return path

    def plot_score_gauge(self, responsible_ai_score):
        """Gauge chart for Responsible AI Score"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=responsible_ai_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Responsible AI Score", 'font': {'size': 24}},
            delta={'reference': 75},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#a78bfa"},
                'steps': [
                    {'range': [0, 40], 'color': "#ff6b6b"},
                    {'range': [40, 60], 'color': "#ffd43b"},
                    {'range': [60, 75], 'color': "#74c0fc"},
                    {'range': [75, 90], 'color': "#51cf66"},
                    {'range': [90, 100], 'color': "#2f9e44"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': responsible_ai_score
                }
            }
        ))

        fig.update_layout(
            height=400,
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            font=dict(color="white")
        )

        path = f"{self.output_dir}/html/score_gauge.html"
        fig.write_html(path)
        print(f"Score gauge saved: {path} ✅")
        return path

    def run(self, shap_results, bias_results,
            compliance_results, responsible_ai_score):
        """Generate all visualizations"""
        print("\n📊 GENERATING ALL VISUALIZATIONS...")
        paths = {}

        # SHAP charts for each sample
        for i in range(len(shap_results)):
            paths[f"shap_{i}"] = self.plot_shap_bar(
                shap_results, row_index=i
            )

        # Fairness dashboard
        paths["fairness"] = self.plot_fairness_dashboard(
            bias_results
        )

        # Compliance heatmap
        paths["compliance"] = self.plot_compliance_heatmap(
            compliance_results
        )

        # Score gauge
        paths["gauge"] = self.plot_score_gauge(
            responsible_ai_score
        )

        print(f"\n{'='*50}")
        print(f"ALL CHARTS GENERATED! ✅")
        print(f"Location: reports/html/")
        print(f"{'='*50}")
        return paths


# ── Run it! ──
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from agents.interpretability_agent import InterpretabilityAgent
    from agents.bias_detector_agent import BiasDetectorAgent
    from orchestrator.compliance_agent import ComplianceAgent

    print("Loading data...")
    X_test = pd.read_csv("data/X_test.csv")
    X_train = pd.read_csv("data/X_train.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    # Run agents
    shap_agent = InterpretabilityAgent()
    shap_agent.load_model("data/credit_model.pkl", "data/feature_names.pkl")
    shap_results = shap_agent.run(X_test, num_samples=3)

    bias_agent = BiasDetectorAgent()
    bias_agent.load_model("data/credit_model.pkl", "data/feature_names.pkl")
    protected_cols = [
        col for col in X_test.columns
        if 'personal_status_sex' in col
    ]
    bias_results = bias_agent.run(X_test, y_test, protected_cols)

    compliance_agent = ComplianceAgent()
    compliance_agent.load_rules("orchestrator/gdpr_rules.json")
    compliance_results = compliance_agent.run(
        domain="credit",
        has_shap=True,
        has_lime=True,
        bias_score=bias_results["fairness_score"],
        feature_count=X_test.shape[1]
    )

    # Generate all visualizations
    viz_agent = VisualizationAgent()
    paths = viz_agent.run(
        shap_results=shap_results,
        bias_results=bias_results,
        compliance_results=compliance_results,
        responsible_ai_score=86
    )

    print("\n🎉 Phase 4 Step 1 Complete!")
    print("Open these files in your browser:")
    for name, path in paths.items():
        print(f"  {name}: {path}")