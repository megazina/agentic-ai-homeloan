import json
import os
from pathlib import Path
import unittest

os.environ["HOME_LOAN_DETERMINISTIC_ONLY"] = "true"

import main


LIKELY_ELIGIBLE = {
    "user_request": "Sensitive applicant text that must not be exported.",
    "gross_annual_income": 180000,
    "monthly_expenses": 4500,
    "deposit": 250000,
    "property_value": 1000000,
    "loan_amount": 750000,
    "employment_type": "permanent",
    "dependants": 1,
    "existing_debts": 20000,
    "residency_status": "citizen",
}

QUALITY_SCENARIO_TARGETS = {
    "hallucination_policy": ("policy", "hallucination"),
    "bias_residency": ("risk_compliance", "bias"),
    "toxicity_applicant": ("conversation_intake", "toxicity"),
    "irrelevant_broker": ("broker_orchestrator", "relevance"),
    "negative_sentiment": ("risk_compliance", "sentiment"),
}


class HomeLoanDeterministicTests(unittest.TestCase):
    def test_likely_eligible_passes_thresholds(self):
        result = main.calculate_eligibility(LIKELY_ELIGIBLE)

        self.assertEqual(result["overall_result"], "PASS")
        self.assertAlmostEqual(result["calculated_values"]["lvr"], 0.75)
        self.assertLess(result["calculated_values"]["dti"], 6.0)
        self.assertTrue(result["checks"]["serviceability"])

    def test_high_lvr_fails_with_reason_code(self):
        payload = {
            **LIKELY_ELIGIBLE,
            "property_value": 1000000,
            "loan_amount": 930000,
        }

        result = main.calculate_eligibility(payload)

        self.assertEqual(result["overall_result"], "FAIL")
        self.assertIn("HIGH_LVR_ABOVE_DEMO_APPETITE", result["reason_codes"])

    def test_high_dti_and_serviceability_fail(self):
        payload = {
            **LIKELY_ELIGIBLE,
            "gross_annual_income": 120000,
            "monthly_expenses": 6200,
            "loan_amount": 750000,
            "existing_debts": 80000,
        }

        result = main.calculate_eligibility(payload)

        self.assertFalse(result["checks"]["dti"])
        self.assertFalse(result["checks"]["serviceability"])
        self.assertIn("DTI_EXCEEDS_THRESHOLD", result["reason_codes"])
        self.assertIn("SERVICEABILITY_SURPLUS_BELOW_THRESHOLD", result["reason_codes"])

    def test_aml_risk_levels(self):
        self.assertEqual(
            main.evaluate_kyc_aml({**LIKELY_ELIGIBLE})["aml_risk_level"],
            "LOW",
        )
        self.assertEqual(
            main.evaluate_kyc_aml({**LIKELY_ELIGIBLE, "residency_status": "visa_holder"})[
                "aml_risk_level"
            ],
            "MEDIUM",
        )
        self.assertEqual(
            main.evaluate_kyc_aml({**LIKELY_ELIGIBLE, "aml_scenario": "high"})[
                "aml_risk_level"
            ],
            "HIGH",
        )

    def test_policy_drift(self):
        aligned = main.evaluate_policy(LIKELY_ELIGIBLE)
        drifted = main.evaluate_policy(
            {
                **LIKELY_ELIGIBLE,
                "policy_version": "HL-POLICY-2025.11",
                "active_policy_version": "HL-POLICY-2026.05",
            }
        )

        self.assertFalse(aligned["policy_drift"])
        self.assertTrue(drifted["policy_drift"])
        self.assertEqual(drifted["drift_status"], "DRIFT")

    def test_final_outcomes(self):
        eligible = main.calculate_eligibility(LIKELY_ELIGIBLE)
        low_risk = {
            "verdict": "PROCEED_AS_DEMO_RECOMMENDATION",
            "disclaimer": "demo",
            "flags": [],
        }

        self.assertEqual(
            main.determine_final_outcome(eligible, low_risk),
            "APPROVE_IN_PRINCIPLE",
        )
        self.assertEqual(
            main.determine_final_outcome(eligible, {"verdict": "ESCALATE"}),
            "REFER",
        )
        self.assertEqual(
            main.determine_final_outcome(eligible, {"verdict": "DECLINE_DEMO_RECOMMENDATION"}),
            "DECLINE",
        )
        self.assertEqual(
            main.determine_final_outcome(eligible, {"verdict": "NEED_MORE_INFO"}),
            "NEED_MORE_INFO",
        )

    def test_quality_issue_scenarios_route_to_target_agents(self):
        all_target_agents = {
            "broker_orchestrator",
            "conversation_intake",
            "kyc_aml",
            "eligibility",
            "policy",
            "risk_compliance",
        }

        for scenario, (target_agent, category) in QUALITY_SCENARIO_TARGETS.items():
            with self.subTest(scenario=scenario):
                state = {
                    "quality_issue_scenario": main._normalise_quality_issue_scenario(
                        scenario
                    )
                }
                issue = main._quality_issue_for_agent(state, target_agent)

                self.assertIsNotNone(issue)
                self.assertEqual(issue["category"], category)
                self.assertIn("Quality issue scenario:", issue["snippet"])

                for other_agent in all_target_agents - {target_agent}:
                    self.assertIsNone(main._quality_issue_for_agent(state, other_agent))

        self.assertIsNone(main._normalise_quality_issue_scenario("unknown_scenario"))


class HomeLoanApiTests(unittest.TestCase):
    def setUp(self):
        self.client = main.app.test_client()

    def test_home_loan_assess_returns_complete_safe_json(self):
        response = self.client.post("/home-loan/assess", json=LIKELY_ELIGIBLE)

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["final_outcome"], "APPROVE_IN_PRINCIPLE")
        self.assertEqual(
            body["agent_path"],
            [
                "A0_BROKER_ORCHESTRATOR",
                "A1_CONVERSATION_INTAKE",
                "A2_KYC_AML",
                "A3_ELIGIBILITY",
                "A4_POLICY",
                "A5_RISK_COMPLIANCE",
                "A6_DECISION_AUDIT",
            ],
        )
        for key in (
            "application_summary",
            "agent_selection_reasons",
            "eligibility_result",
            "policy_result",
            "risk_compliance_result",
            "final_outcome",
            "audit_record",
        ):
            self.assertIn(key, body)

        serialized = json.dumps(body)
        self.assertNotIn(LIKELY_ELIGIBLE["user_request"], serialized)
        self.assertFalse(body["audit_record"]["redaction"]["raw_prompt_exported"])

    def test_sample_payloads_execute_expected_paths(self):
        sample_dir = Path(__file__).resolve().parents[1] / "sample_payloads"
        expected_outcomes = {
            "likely_eligible.json": "APPROVE_IN_PRINCIPLE",
            "high_lvr.json": "DECLINE",
            "high_dti_serviceability_fail.json": "REFER",
            "aml_escalation.json": "REFER",
            "policy_drift.json": "REFER",
            "quality_hallucination_policy.json": "REFER",
            "quality_bias_residency.json": "APPROVE_IN_PRINCIPLE",
            "quality_toxicity_applicant.json": "APPROVE_IN_PRINCIPLE",
            "quality_irrelevant_broker.json": "APPROVE_IN_PRINCIPLE",
            "quality_negative_sentiment.json": "REFER",
        }

        for filename, expected_outcome in expected_outcomes.items():
            with self.subTest(filename=filename):
                payload = json.loads((sample_dir / filename).read_text())
                response = self.client.post("/home-loan/assess", json=payload)
                body = response.get_json()

                self.assertEqual(response.status_code, 200)
                self.assertEqual(body["final_outcome"], expected_outcome)
                self.assertEqual(len(body["agent_steps"]), 7)

                serialized = json.dumps(body)
                self.assertNotIn("Quality issue scenario:", serialized)
                if payload.get("quality_issue_scenario"):
                    self.assertNotIn(payload["quality_issue_scenario"], serialized)


if __name__ == "__main__":
    unittest.main()
