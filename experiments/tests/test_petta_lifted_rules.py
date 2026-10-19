import importlib.util
import os
import sys
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.services.petta_service import PeTTaService


JANUS_AVAILABLE = importlib.util.find_spec("janus_swi") is not None


@unittest.skipUnless(JANUS_AVAILABLE, "janus_swi is required for PeTTa integration tests")
class TestPeTTaLiftedRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        project_root_metta = PROJECT_ROOT.replace("\\", "/")
        setup_metta = f"""
!(import! &self {project_root_metta}/experiments/utils/common-utils)
!(import! &self {project_root_metta}/experiments/frequent-pattern-miner/etv-utils)
"""
        cls.service = PeTTaService.create_required(PROJECT_ROOT, setup_metta, verbose=False)

    def test_direct_fact_query_uncurries_partial_output(self) -> None:
        self.assertIsNotNone(self.service.add_atom('(: fact_length_bucket (length-bucket A_known "Medium") (STV 0.694 0.9))'))

        proofs = self.service.query('(: $prf (length-bucket A_known "Medium") $tv)', depth=10)

        self.assertTrue(proofs)
        self.assertTrue(any('(length-bucket A_known "Medium")' in proof for proof in proofs))
        self.assertFalse(any('(partial length-bucket (A_known "Medium"))' in proof for proof in proofs))

    def test_lifted_forward_rule_answers_grounded_query(self) -> None:
        self.assertIsNotNone(self.service.add_atom('(: fact_size (size_bucket A_known "Low") (STV 1.0 1.0))'))
        self.assertIsNotNone(self.service.add_atom('(: fact_tone (tone_bucket A_known "Analytical") (STV 1.0 1.0))'))
        self.assertIsNotNone(
            self.service.add_forward_only_rule(
                '(: rule_1 (-> (And (size_bucket $x "Low") (tone_bucket $x "Analytical")) '
                '(engagement $x "High")) (STV 0.62 0.81))'
            )
        )

        grounded_proofs = self.service.query('(: $prf (engagement A_known "High") $tv)', depth=10)
        open_proofs = self.service.query('(: $prf (engagement $article "High") $tv)', depth=10)

        self.assertTrue(grounded_proofs)
        self.assertTrue(any('(engagement A_known "High")' in proof for proof in grounded_proofs))
        self.assertTrue(any('rule_1' in proof for proof in grounded_proofs))
        self.assertTrue(any('(engagement A_known "High")' in proof for proof in open_proofs))

    def test_direct_and_multiple_rule_proofs_are_all_returned(self) -> None:
        self.assertIsNotNone(self.service.add_atom('(: fact_engagement (engagement A_multi "High") (STV 0.91 0.96))'))
        self.assertIsNotNone(self.service.add_atom('(: fact_tone_multi (tone_bucket A_multi "Analytical") (STV 1.0 1.0))'))
        self.assertIsNotNone(self.service.add_atom('(: fact_size_multi (size_bucket A_multi "Low") (STV 1.0 1.0))'))
        self.assertIsNotNone(self.service.add_atom('(: fact_topic_multi (topic A_multi "AI") (STV 1.0 1.0))'))
        self.assertIsNotNone(
            self.service.add_forward_only_rule(
                '(: rule_multi_1 (-> (And (tone_bucket $x "Analytical") (size_bucket $x "Low")) '
                '(engagement $x "High")) (STV 0.62 0.81))'
            )
        )
        self.assertIsNotNone(
            self.service.add_forward_only_rule(
                '(: rule_multi_2 (-> (And (tone_bucket $x "Analytical") (topic $x "AI")) '
                '(engagement $x "High")) (STV 0.7 0.8))'
            )
        )

        proofs = self.service.query('(: $prf (engagement A_multi "High") $tv)', depth=10)

        self.assertGreaterEqual(len(proofs), 3)
        self.assertTrue(any('fact_engagement' in proof for proof in proofs))
        self.assertTrue(any('rule_multi_1' in proof for proof in proofs))
        self.assertTrue(any('rule_multi_2' in proof for proof in proofs))


if __name__ == "__main__":
    unittest.main()
