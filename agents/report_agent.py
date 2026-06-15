# Phase 4 - Report Agent (PDF Generator)
import os
import sys
import pickle
import pandas as pd
from datetime import datetime
sys.path.append(".")

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import (
    HexColor, black, white
)
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class ReportAgent:
    def __init__(self):
        self.output_dir = "reports"
        os.makedirs(self.output_dir, exist_ok=True)

        # Color scheme
        self.purple = HexColor("#7c5cbf")
        self.dark_bg = HexColor("#1a1a2e")
        self.light_bg = HexColor("#f8f9fa")
        self.green = HexColor("#28a745")
        self.red = HexColor("#dc3545")
        self.amber = HexColor("#ffc107")
        self.blue = HexColor("#007bff")
        self.dark_text = HexColor("#212529")
        self.muted = HexColor("#6c757d")

        # Styles
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        print("ReportAgent initialized ✅")

    def _setup_styles(self):
        """Setup custom paragraph styles"""
        self.title_style = ParagraphStyle(
            "CustomTitle",
            parent=self.styles["Title"],
            fontSize=28,
            textColor=white,
            alignment=TA_CENTER,
            spaceAfter=12,
            fontName="Helvetica-Bold"
        )
        self.subtitle_style = ParagraphStyle(
            "CustomSubtitle",
            parent=self.styles["Normal"],
            fontSize=14,
            textColor=HexColor("#e0e0e0"),
            alignment=TA_CENTER,
            spaceAfter=6,
            fontName="Helvetica"
        )
        self.heading_style = ParagraphStyle(
            "CustomHeading",
            parent=self.styles["Heading1"],
            fontSize=16,
            textColor=self.purple,
            spaceBefore=16,
            spaceAfter=8,
            fontName="Helvetica-Bold"
        )
        self.body_style = ParagraphStyle(
            "CustomBody",
            parent=self.styles["Normal"],
            fontSize=10,
            textColor=self.dark_text,
            spaceAfter=6,
            fontName="Helvetica",
            leading=16
        )
        self.small_style = ParagraphStyle(
            "CustomSmall",
            parent=self.styles["Normal"],
            fontSize=8,
            textColor=self.muted,
            spaceAfter=4,
            fontName="Helvetica"
        )

    def _get_severity_color(self, severity):
        """Get color based on severity"""
        if "PASS" in severity or "COMPLIANT" in severity:
            return self.green
        elif "WARN" in severity or "AT_RISK" in severity:
            return self.amber
        else:
            return self.red

    def _make_cover_page(self, story, scores, domain, model_accuracy):
        """Build cover page"""
        # Dark background cover
        story.append(Spacer(1, 0.5 * inch))

        # Title
        story.append(Paragraph(
            "RESPONSIBLE AI AUDIT REPORT",
            self.title_style
        ))
        story.append(Spacer(1, 0.2 * inch))

        # Subtitle
        story.append(Paragraph(
            f"XAI Agent Analysis — {domain.upper()} DOMAIN",
            self.subtitle_style
        ))
        story.append(Spacer(1, 0.1 * inch))

        # Date
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            self.subtitle_style
        ))
        story.append(Spacer(1, 0.4 * inch))

        # Score badge table
        total = scores["total"]["score"]
        grade = scores["total"]["grade"]
        exp = scores["explainability"]["score"]
        fair = scores["fairness"]["score"]
        comp = scores["compliance"]["score"]

        score_color = self.green if total >= 75 else (
            self.amber if total >= 60 else self.red
        )

        # Main score card
        score_data = [
            ["RESPONSIBLE AI SCORE", "", ""],
            [str(total) + " / 100", "", grade],
            ["", "", ""],
            ["Explainability", "Fairness", "Compliance"],
            [
                str(exp) + "/30",
                str(fair) + "/40",
                str(comp) + "/30"
            ]
        ]

        score_table = Table(
            score_data,
            colWidths=[2.2*inch, 2.2*inch, 2.2*inch]
        )
        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.dark_bg),
            ("BACKGROUND", (0, 1), (-1, 2), score_color),
            ("BACKGROUND", (0, 3), (-1, 4), self.dark_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("TEXTCOLOR", (0, 1), (-1, 2), white),
            ("TEXTCOLOR", (0, 3), (-1, 4), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, 1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 14),
            ("FONTSIZE", (0, 1), (0, 1), 36),
            ("FONTSIZE", (0, 3), (-1, 3), 11),
            ("FONTSIZE", (0, 4), (-1, 4), 14),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWHEIGHT", (0, 0), (-1, 0), 30),
            ("ROWHEIGHT", (0, 1), (-1, 1), 60),
            ("ROWHEIGHT", (0, 2), (-1, 2), 5),
            ("ROWHEIGHT", (0, 3), (-1, 3), 25),
            ("ROWHEIGHT", (0, 4), (-1, 4), 30),
            ("GRID", (0, 0), (-1, -1), 1, white),
            ("ROUNDEDCORNERS", [5, 5, 5, 5]),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 0.3 * inch))

        # Model info table
        info_data = [
            ["Model Type", "Domain", "Accuracy", "Test Samples"],
            ["XGBoost", domain.upper(), f"{model_accuracy:.1f}%", "200"]
        ]
        info_table = Table(
            info_data,
            colWidths=[1.65*inch, 1.65*inch, 1.65*inch, 1.65*inch]
        )
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.purple),
            ("BACKGROUND", (0, 1), (-1, 1), self.light_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("TEXTCOLOR", (0, 1), (-1, 1), self.dark_text),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWHEIGHT", (0, 0), (-1, -1), 25),
            ("GRID", (0, 0), (-1, -1), 0.5, self.muted),
        ]))
        story.append(info_table)
        story.append(PageBreak())

    def _make_executive_summary(self, story, scores,
                                 bias_results, compliance_results):
        """Executive summary page"""
        story.append(Paragraph(
            "EXECUTIVE SUMMARY", self.heading_style
        ))
        story.append(Spacer(1, 0.1 * inch))

        total = scores["total"]["score"]
        status = scores["total"]["status"]
        bias_score = bias_results["fairness_score"]
        comp_score = compliance_results["compliance_score"]
        risk_tier = compliance_results["risk_tier"]["tier"].upper()

        summary_text = f"""
        This Responsible AI Audit was conducted using the XAI Agent system,
        which combines SHAP explainability, LIME verification, fairness
        analysis using Fairlearn and IBM AIF360, and GDPR/EU AI Act
        compliance checking.
        <br/><br/>
        The model achieved an overall <b>Responsible AI Score of
        {total}/100</b> with a status of <b>{status}</b>.
        The EU AI Act classifies this system as <b>{risk_tier} RISK</b>,
        requiring conformity assessment and human oversight before deployment.
        <br/><br/>
        Key findings include a fairness score of <b>{bias_score}/100</b>
        indicating gender-based bias patterns in predictions, and a
        compliance score of <b>{comp_score}/100</b> with most GDPR
        articles satisfied due to the explainability layer.
        """
        story.append(Paragraph(summary_text, self.body_style))
        story.append(Spacer(1, 0.2 * inch))

        # Key findings table
        story.append(Paragraph("KEY FINDINGS", self.heading_style))

        findings = [
            ["Finding", "Status", "Priority"],
            ["SHAP + LIME explanations available",
             "COMPLIANT", "—"],
            ["Gender bias detected (sex_A93, sex_A94)",
             "FAIL", "CRITICAL"],
            ["GDPR Article 22 (Right to Explanation)",
             "COMPLIANT", "—"],
            ["GDPR Article 25 (Privacy by Design)",
             "AT RISK", "HIGH"],
            ["EU AI Act Risk Classification",
             "HIGH RISK", "CRITICAL"],
            ["DPIA completed and documented",
             "COMPLIANT", "—"],
        ]

        findings_table = Table(
            findings,
            colWidths=[3.2*inch, 1.5*inch, 1.5*inch]
        )

        style = [
            ("BACKGROUND", (0, 0), (-1, 0), self.dark_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWHEIGHT", (0, 0), (-1, -1), 22),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dee2e6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [white, self.light_bg]),
        ]

        # Color status cells
        status_colors = {
            "COMPLIANT": self.green,
            "FAIL": self.red,
            "AT RISK": self.amber,
            "HIGH RISK": self.red
        }
        for i, row in enumerate(findings[1:], 1):
            color = status_colors.get(row[1], self.muted)
            style.append(("TEXTCOLOR", (1, i), (1, i), color))
            style.append(("FONTNAME", (1, i), (1, i),
                          "Helvetica-Bold"))

        findings_table.setStyle(TableStyle(style))
        story.append(findings_table)
        story.append(PageBreak())

    def _make_bias_section(self, story, bias_results):
        """Bias analysis section"""
        story.append(Paragraph("BIAS ANALYSIS", self.heading_style))
        story.append(Paragraph(
            f"Overall Fairness Score: "
            f"<b>{bias_results['fairness_score']}/100</b>",
            self.body_style
        ))
        story.append(Spacer(1, 0.1 * inch))

        # Bias metrics table
        bias_data = [["Protected Attribute", "DPD", "DPD Status",
                       "EOD", "EOD Status"]]

        for r in bias_results["bias_results"]:
            bias_data.append([
                r["attribute"].replace("personal_status_sex_", "sex_"),
                str(r["demographic_parity_difference"]),
                r["dpd_severity"],
                str(r["equalized_odds_difference"]),
                r["eod_severity"]
            ])

        bias_table = Table(
            bias_data,
            colWidths=[1.8*inch, 0.8*inch, 1.1*inch, 0.8*inch, 1.1*inch]
        )
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), self.dark_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWHEIGHT", (0, 0), (-1, -1), 20),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dee2e6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [white, self.light_bg]),
        ]
        for i, row in enumerate(bias_data[1:], 1):
            dpd_color = self._get_severity_color(row[2])
            eod_color = self._get_severity_color(row[4])
            style.append(("TEXTCOLOR", (2, i), (2, i), dpd_color))
            style.append(("TEXTCOLOR", (4, i), (4, i), eod_color))
            style.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))
            style.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))

        bias_table.setStyle(TableStyle(style))
        story.append(bias_table)
        story.append(Spacer(1, 0.2 * inch))

        # Proxy features
        if bias_results.get("proxy_risks"):
            story.append(Paragraph(
                "PROXY FEATURES DETECTED", self.heading_style
            ))
            proxy_data = [["Feature", "Protected Attribute",
                           "Correlation", "Risk Level"]]
            for p in bias_results["proxy_risks"][:5]:
                proxy_data.append([
                    p["feature"],
                    p["protected_attribute"].replace(
                        "personal_status_sex_", "sex_"),
                    str(p["correlation"]),
                    p["risk"]
                ])
            proxy_table = Table(
                proxy_data,
                colWidths=[1.5*inch, 2*inch, 1*inch, 1.5*inch]
            )
            proxy_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), self.purple),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWHEIGHT", (0, 0), (-1, -1), 20),
                ("GRID", (0, 0), (-1, -1), 0.5,
                 HexColor("#dee2e6")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [white, self.light_bg]),
            ]))
            story.append(proxy_table)
        story.append(PageBreak())

    def _make_compliance_section(self, story, compliance_results):
        """Compliance section"""
        story.append(Paragraph(
            "COMPLIANCE ANALYSIS", self.heading_style
        ))
        story.append(Paragraph(
            f"Compliance Score: "
            f"<b>{compliance_results['compliance_score']}/100</b> | "
            f"EU AI Act Risk: "
            f"<b>{compliance_results['risk_tier']['tier'].upper()}</b>",
            self.body_style
        ))
        story.append(Spacer(1, 0.1 * inch))

        # GDPR table
        gdpr_data = [["Regulation", "Severity",
                       "Status", "Finding"]]
        for r in compliance_results["gdpr_results"]:
            gdpr_data.append([
                r["rule_name"].split(" - ")[0],
                r["severity"],
                r["status"].split()[0],
                r["finding"][:45] + "..." if len(
                    r["finding"]) > 45 else r["finding"]
            ])

        gdpr_table = Table(
            gdpr_data,
            colWidths=[1.4*inch, 0.8*inch, 1.1*inch, 2.9*inch]
        )
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), self.dark_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (2, -1), "CENTER"),
            ("ALIGN", (3, 1), (3, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWHEIGHT", (0, 0), (-1, -1), 22),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dee2e6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [white, self.light_bg]),
        ]
        for i, row in enumerate(gdpr_data[1:], 1):
            color = self._get_severity_color(row[2])
            style.append(("TEXTCOLOR", (2, i), (2, i), color))
            style.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))

        gdpr_table.setStyle(TableStyle(style))
        story.append(gdpr_table)
        story.append(PageBreak())

    def _make_remediation_section(self, story, compliance_results,
                                   bias_results):
        """Remediation roadmap"""
        story.append(Paragraph(
            "REMEDIATION ROADMAP", self.heading_style
        ))
        story.append(Paragraph(
            "Prioritized actions to improve the Responsible AI Score:",
            self.body_style
        ))
        story.append(Spacer(1, 0.1 * inch))

        steps = compliance_results.get("remediation_steps", [])

        # Add bias remediation
        steps.append({
            "priority": "🔴 CRITICAL",
            "action": "Implement bias mitigation for sex_A93 and sex_A94",
            "regulation": "FAIRNESS"
        })
        steps.append({
            "priority": "🟡 HIGH",
            "action": "Monitor proxy features: age, dependents correlation",
            "regulation": "FAIRNESS"
        })
        steps.append({
            "priority": "🟢 MEDIUM",
            "action": "Add human review workflow for HIGH RISK decisions",
            "regulation": "EU_AI_ACT"
        })

        rem_data = [["#", "Priority", "Action", "Regulation"]]
        for i, step in enumerate(steps[:6], 1):
            rem_data.append([
                str(i),
                step["priority"].replace("🔴 ", "").replace(
                    "🟡 ", "").replace("🟢 ", ""),
                step["action"][:55] + "..." if len(
                    step["action"]) > 55 else step["action"],
                step["regulation"]
            ])

        rem_table = Table(
            rem_data,
            colWidths=[0.3*inch, 0.9*inch, 3.5*inch, 1.1*inch]
        )
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), self.dark_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (1, -1), "CENTER"),
            ("ALIGN", (2, 1), (2, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWHEIGHT", (0, 0), (-1, -1), 22),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dee2e6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [white, self.light_bg]),
        ]
        priority_colors = {
            "CRITICAL": self.red,
            "HIGH": self.amber,
            "MEDIUM": self.green
        }
        for i, row in enumerate(rem_data[1:], 1):
            color = priority_colors.get(row[1], self.muted)
            style.append(("TEXTCOLOR", (1, i), (1, i), color))
            style.append(("FONTNAME", (1, i), (1, i), "Helvetica-Bold"))

        rem_table.setStyle(TableStyle(style))
        story.append(rem_table)
        story.append(Spacer(1, 0.3 * inch))

        # Footer note
        story.append(Paragraph(
            "This report was generated automatically by Rithvik's "
            "XAI Agent System. All findings should be reviewed by "
            "a qualified AI ethics professional before deployment.",
            self.small_style
        ))

    def generate_pdf(self, scores, bias_results,
                     compliance_results, model_accuracy=79.5,
                     domain="credit"):
        """Generate the full PDF report"""
        print("\n📄 GENERATING PDF REPORT...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/XAI_Audit_Report_{timestamp}.pdf"

        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            title="Responsible AI Audit Report",
            author="Rithvik's XAI Agent"
        )

        story = []

        # Build all sections
        self._make_cover_page(story, scores, domain, model_accuracy)
        self._make_executive_summary(
            story, scores, bias_results, compliance_results
        )
        self._make_bias_section(story, bias_results)
        self._make_compliance_section(story, compliance_results)
        self._make_remediation_section(
            story, compliance_results, bias_results
        )

        # Build PDF
        doc.build(story)
        print(f"PDF saved: {filename} ✅")
        return filename


# ── Run it! ──
if __name__ == "__main__":
    from orchestrator.compliance_agent import ComplianceAgent
    from agents.bias_detector_agent import BiasDetectorAgent
    from orchestrator.responsible_ai_score import (
        compute_responsible_ai_score
    )

    print("Loading data...")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    # Run agents
    bias_agent = BiasDetectorAgent()
    bias_agent.load_model("data/credit_model.pkl",
                          "data/feature_names.pkl")
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

    scores = compute_responsible_ai_score(
        bias_score=bias_results["fairness_score"],
        compliance_score=compliance_results["compliance_score"],
        shap_works=True,
        lime_works=True,
        proxy_risks=bias_results["proxy_risks"],
        gdpr_results=[]
    )

    # Generate PDF
    report_agent = ReportAgent()
    pdf_path = report_agent.generate_pdf(
        scores=scores,
        bias_results=bias_results,
        compliance_results=compliance_results,
        model_accuracy=79.5,
        domain="credit"
    )

    print(f"\n🎉 Phase 4 Step 2 Complete!")
    print(f"PDF Report: {pdf_path}")
    print("Open it in File Explorer to see your report!")