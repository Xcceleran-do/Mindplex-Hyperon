import unittest
from unittest.mock import patch

try:
    from experiments import mining_api
    from experiments.api import runtime as petta_runtime
    from experiments.api import simulation as simulation_service
    MINING_API_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-specific optional deps
    mining_api = None
    petta_runtime = None
    simulation_service = None
    MINING_API_IMPORT_ERROR = exc


@unittest.skipUnless(mining_api is not None, f"mining_api dependencies unavailable: {MINING_API_IMPORT_ERROR}")
class TestSimulationEngine(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_dataset_facts = list(petta_runtime.chainer_dataset_facts)
        self._saved_rule_atoms = list(petta_runtime.chainer_rule_atoms)
        petta_runtime.chainer_dataset_facts = []
        petta_runtime.chainer_rule_atoms = []

    def tearDown(self) -> None:
        petta_runtime.chainer_dataset_facts = self._saved_dataset_facts
        petta_runtime.chainer_rule_atoms = self._saved_rule_atoms

    def test_build_simulation_fact_atoms_uses_active_legacy_length_predicate(self) -> None:
        petta_runtime.chainer_dataset_facts = [
            '(: (fact:- (length A_1 "Medium")) (length A_1 "Medium") (STV 0.7 0.9))'
        ]

        facts = simulation_service.build_simulation_fact_atoms(
            {"attributes": {"length": "Medium", "tone": "Analytical"}},
            "H_sim_1",
        )

        self.assertTrue(any('(length H_sim_1 "Medium")' in fact for fact in facts))
        self.assertFalse(any('(length-bucket H_sim_1 "Medium")' in fact for fact in facts))
        self.assertTrue(any('(tone H_sim_1 "Analytical")' in fact for fact in facts))

    def test_build_simulation_fact_atoms_uses_length_bucket_when_active(self) -> None:
        petta_runtime.chainer_rule_atoms = [
            '(: rule_1 (-> (length-bucket $x "Medium") (engagement $x "High")) (STV 0.7 0.8))'
        ]

        facts = simulation_service.build_simulation_fact_atoms(
            {"attributes": {"length": "Medium"}},
            "H_sim_2",
        )

        self.assertTrue(any('(length-bucket H_sim_2 "Medium")' in fact for fact in facts))

    def test_simulate_engagement_aggregates_bucket_scores(self) -> None:
        petta_runtime.chainer_rule_atoms = [
            '(: rule_1 (-> (tone $x "Analytical") (engagement $x "High")) (STV 0.8 0.9))'
        ]
        petta_runtime.chainer_dataset_facts = [
            '(: (fact:- (engagement A_1 "High")) (engagement A_1 "High") (STV 0.8 0.9))',
            '(: (fact:- (engagement A_2 "Low")) (engagement A_2 "Low") (STV 0.2 0.9))',
        ]

        worker_result = {
            "status": "success",
            "proofs_by_level": {
                "High": ['(: p1 (engagement H_sim_1 "High") (STV 0.8 0.9))'],
                "Medium": [],
                "Low": ['(: p2 (engagement H_sim_1 "Low") (STV 0.2 0.9))'],
            },
        }

        with patch.object(simulation_service, "reload_petta_dataset_if_ready", return_value={"status": "unchanged"}), \
             patch.object(simulation_service, "run_simulation_worker", return_value=worker_result):
            result = simulation_service.simulate_engagement(
                {"article_id": "sim_1", "attributes": {"tone": "Analytical"}, "depth": 2}
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["predicted_engagement"], "High")
        self.assertFalse(result["used_prior_fallback"])
        self.assertGreater(result["probabilities"]["High"], result["probabilities"]["Low"])
        self.assertEqual(result["buckets"]["High"]["proof_count"], 1)
        self.assertEqual(result["buckets"]["Medium"]["proof_count"], 0)
        self.assertIn("explanation", result)
        self.assertEqual(result["explanation"]["summary"], "At least one mined rule fired; probabilities were normalized from proof STVs.")

    def test_build_simulation_explanation_maps_proof_to_rule_and_fact(self) -> None:
        explanation = simulation_service.build_simulation_explanation(
            article_id="H_demo",
            hypothetical_facts=[
                '(: sim_fact_1 (audience-expertise H_demo "Intermediate") (STV 1.0 1.0))'
            ],
            rule_atoms=[
                '(: rule_1 (-> (audience-expertise $x "Intermediate") (engagement $x "Low")) (STV 0.6 0.8))'
            ],
            proofs_by_level={
                "High": [],
                "Medium": [],
                "Low": ['(: (rule_1 sim_fact_1) (engagement H_demo "Low") (STV 0.6 0.8))'],
            },
            used_prior_fallback=False,
        )

        low_chain = explanation["chains_by_level"]["Low"][0]
        self.assertEqual(low_chain["rule_id"], "rule_1")
        self.assertEqual(low_chain["rule"]["consequent"]["value"], "Low")
        self.assertEqual(low_chain["facts"][0]["id"], "sim_fact_1")
        self.assertEqual(low_chain["stv"], {"strength": 0.6, "confidence": 0.8})

    def test_build_simulation_explanation_reports_missing_antecedents_for_fallback(self) -> None:
        explanation = simulation_service.build_simulation_explanation(
            article_id="H_demo",
            hypothetical_facts=[
                '(: sim_fact_1 (audience-expertise H_demo "Beginner") (STV 1.0 1.0))'
            ],
            rule_atoms=[
                '(: rule_1 (-> (audience-expertise $x "Intermediate") (engagement $x "Low")) (STV 0.6 0.8))'
            ],
            proofs_by_level={"High": [], "Medium": [], "Low": []},
            used_prior_fallback=True,
        )

        self.assertIn("historical engagement priors", explanation["summary"])
        self.assertEqual(
            explanation["unmatched_rules"][0]["missing_antecedents"],
            ['(audience-expertise H_demo "Intermediate")'],
        )


if __name__ == "__main__":
    unittest.main()
