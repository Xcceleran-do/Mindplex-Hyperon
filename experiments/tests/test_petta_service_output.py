import unittest

from experiments.services.petta_service import normalize_chainer_result, normalize_partial_results


class TestPeTTaServiceOutput(unittest.TestCase):
    def test_normalize_partial_results_uncurries_direct_fact(self) -> None:
        proof = '(: (fact:- (partial length (A_16624 "Medium"))) (partial length (A_16624 "Medium")) (STV 0.694 0.9))'

        self.assertEqual(
            normalize_partial_results(proof),
            '(: (fact:- (length A_16624 "Medium")) (length A_16624 "Medium") (STV 0.694 0.9))',
        )

    def test_normalize_chainer_result_uncurries_nested_partial_premises(self) -> None:
        proof = (
            '(: proof_1 (-> (And (partial length (A_1 "Low")) (topic A_1 "AI")) '
            '(engagement A_1 "High")) (STV 0.5 0.6))'
        )

        self.assertEqual(
            normalize_chainer_result([proof]),
            [
                '(: proof_1 (-> (And (length A_1 "Low") (topic A_1 "AI")) '
                '(engagement A_1 "High")) (STV 0.5 0.6))'
            ],
        )


if __name__ == "__main__":
    unittest.main()
