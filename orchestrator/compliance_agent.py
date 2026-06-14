# Phase 3 - Compliance Agent
import json
import os
import sys
sys.path.append(".")

class ComplianceAgent:
    def __init__(self):
        self.rules = None
        self.eu_ai_act = None
        print("ComplianceAgent initialized ✅")

    def load_rules(self, rules_path):
        with open(rules_path, "r") as f:
            data = json.load(f)
        self.rules = data["regulations"]
        self.eu_ai_act = data["eu_ai_act"]
        print(f"Loaded {len(self.rules)} GDPR rules ✅")

    def get_eu_risk_tier(self, domain):
        print(f"\n{'='*50}")
        print(f"EU AI ACT RISK CLASSIFICATION")
        print(f"{'='*50}")
        print(f"Domain: {domain}")

        tiers = self.eu_ai_act["risk_tiers"]
        for tier_name, tier_data in tiers.items():
            if domain.lower() in tier_data["domains"]:
                print(f"\nRisk Tier: {tier_name.upper()} ⚠️")
                print(f"Action Required: {tier_data['action']}")
                return {
                    "tier": tier_name,
                    "action": tier_data["action"],
                    "color": tier_data["color"]
                }

        print(f"\nRisk Tier: LIMITED (default)")
        return {
            "tier": "limited",
            "action": tiers["limited"]["action"],
            "color": tiers["limited"]["color"]
        }

    def check_gdpr(self, has_shap, has_lime, bias_score,
                   feature_count, domain):
        print(f"\n{'='*50}")
        print(f"GDPR COMPLIANCE CHECKS")
        print(f"{'='*50}")

        results = []
        for rule in self.rules:
            check = rule["check"]
            status = "COMPLIANT ✅"
            finding = ""
            remediation = ""

            if check == "has_explanation":
                if has_shap and has_lime:
                    status = "COMPLIANT ✅"
                    finding = "SHAP and LIME explanations available"
                elif has_shap or has_lime:
                    status = "PARTIAL ⚠️"
                    finding = "Only one explanation method available"
                    remediation = "Add both SHAP and LIME explanations"
                else:
                    status = "NON_COMPLIANT ❌"
                    finding = "No explanation method found"
                    remediation = "Implement SHAP or LIME explanations"

            elif check == "data_minimization":
                if feature_count > 50:
                    status = "AT_RISK ⚠️"
                    finding = f"{feature_count} features may be excessive"
                    remediation = "Review and remove unnecessary features"
                else:
                    status = "COMPLIANT ✅"
                    finding = f"{feature_count} features is acceptable"

            elif check == "transparency":
                if has_shap or has_lime:
                    status = "COMPLIANT ✅"
                    finding = "Model decisions can be explained"
                else:
                    status = "NON_COMPLIANT ❌"
                    finding = "Model is a black box"
                    remediation = "Add explainability layer"

            elif check == "privacy_by_design":
                if bias_score < 50:
                    status = "AT_RISK ⚠️"
                    finding = "Low fairness score indicates privacy risks"
                    remediation = "Address bias before deployment"
                else:
                    status = "COMPLIANT ✅"
                    finding = "Fairness score acceptable"

            elif check == "impact_assessment":
                if domain in ["credit", "healthcare", "employment"]:
                    dpia_exists = os.path.exists(
                        "orchestrator/dpia.json"
                    )
                    if dpia_exists:
                        status = "COMPLIANT ✅"
                        finding = "DPIA completed and documented"
                    else:
                        status = "REQUIRED ⚠️"
                        finding = "DPIA required for this domain"
                        remediation = "Conduct Data Protection Impact Assessment"
                else:
                    status = "COMPLIANT ✅"
                    finding = "DPIA not required for this domain"

            print(f"\n{rule['name']}")
            print(f"  Status: {status}")
            print(f"  Finding: {finding}")
            if remediation:
                print(f"  Fix: {remediation}")

            results.append({
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "severity": rule["severity"],
                "status": status,
                "finding": finding,
                "remediation": remediation
            })

        return results

    def compute_compliance_score(self, gdpr_results, risk_tier):
        score = 100

        for r in gdpr_results:
            if "NON_COMPLIANT" in r["status"]:
                if r["severity"] == "CRITICAL":
                    score -= 25
                else:
                    score -= 15
            elif "AT_RISK" in r["status"] or "PARTIAL" in r["status"]:
                if r["severity"] == "CRITICAL":
                    score -= 15
                else:
                    score -= 8
            elif "REQUIRED" in r["status"]:
                score -= 10

        if risk_tier["tier"] == "high":
            score -= 10
        elif risk_tier["tier"] == "unacceptable":
            score = 0

        return max(0, score)

    def get_remediation_steps(self, gdpr_results, risk_tier):
        steps = []

        for r in gdpr_results:
            if r["remediation"]:
                priority = "🔴 CRITICAL" if r["severity"] == "CRITICAL" else "🟡 HIGH"
                steps.append({
                    "priority": priority,
                    "action": r["remediation"],
                    "regulation": r["rule_id"]
                })

        if risk_tier["tier"] == "high":
            steps.append({
                "priority": "🔴 CRITICAL",
                "action": risk_tier["action"],
                "regulation": "EU_AI_ACT"
            })

        return steps

    def run(self, domain, has_shap, has_lime,
            bias_score, feature_count):
        print("\n📜 STARTING COMPLIANCE AUDIT...")

        risk_tier = self.get_eu_risk_tier(domain)

        gdpr_results = self.check_gdpr(
            has_shap, has_lime,
            bias_score, feature_count, domain
        )

        compliance_score = self.compute_compliance_score(
            gdpr_results, risk_tier
        )

        remediation = self.get_remediation_steps(
            gdpr_results, risk_tier
        )

        print(f"\n{'='*50}")
        print(f"COMPLIANCE SCORE: {compliance_score}/100")
        if compliance_score >= 80:
            print("Status: COMPLIANT ✅")
        elif compliance_score >= 60:
            print("Status: NEEDS WORK ⚠️")
        else:
            print("Status: NON-COMPLIANT ❌")
        print(f"{'='*50}")

        if remediation:
            print(f"\nTOP ACTIONS TO TAKE:")
            for i, step in enumerate(remediation[:3]):
                print(f"{i+1}. {step['priority']}: {step['action']}")

        return {
            "domain": domain,
            "risk_tier": risk_tier,
            "gdpr_results": gdpr_results,
            "compliance_score": compliance_score,
            "remediation_steps": remediation
        }


# ── Run it! ──
if __name__ == "__main__":
    agent = ComplianceAgent()
    agent.load_rules("orchestrator/gdpr_rules.json")

    results = agent.run(
        domain="credit",
        has_shap=True,
        has_lime=True,
        bias_score=10,
        feature_count=48
    )

    print("\n" + "="*50)
    print("Phase 3 Step 1 Complete!")
    print(f"Compliance Score: {results['compliance_score']}/100")
    print(f"EU AI Act Tier: {results['risk_tier']['tier'].upper()}")
    print("="*50)