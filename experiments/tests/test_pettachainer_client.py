import unittest

from experiments.api.support import patterns_to_chainer_rules


class TestPeTTaChainerFormatting(unittest.TestCase):
    def test_mined_pattern_uses_server_rule_shape(self) -> None:
        rules = patterns_to_chainer_rules(
            [
                {
                    "pattern": (
                        'supportOf ((And (size_bucket $_123 "Low") '
                        '(tone_bucket $_123 "Analytical") '
                        '(engagement $_123 "High")) (STV 0.62 0.81)) 4'
                    )
                }
            ]
        )
        self.assertEqual(
            rules,
            [
                '(: rule_1 (Implication (Premises (size_bucket $x "Low") '
                '(tone_bucket $x "Analytical")) (Conclusions (engagement $x "High"))) '
                '(STV 0.62 0.81))'
            ],
        )


if __name__ == "__main__":
    unittest.main()
