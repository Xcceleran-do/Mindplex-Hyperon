import unittest

from experiments.api.support import patterns_to_chainer_rules


class TestPeTTaChainerFormatting(unittest.TestCase):
    def test_mined_pattern_uses_server_rule_shape(self) -> None:
        rules = patterns_to_chainer_rules(
            [   {'pattern': '((audience-expertise $_14232 "intermediate") (engagement $_14232 "High")) (STV 0.13333333333333333 0.8571428571428571)', 'support': '6'}, 
                {'pattern': '((audience-expertise $_14360 "intermediate") (engagement $_14360 "Very_High")) (STV 0.7777777777777778 0.9722222222222222)', 'support': '35'}
            ])
        self.assertEqual(
            rules,
            [
                '((: rule_1 (Implication (Premises (audience-expertise $_14232 "intermediate")) (Conclusions (engagement $_14232 "High"))) (STV 0.13333333333333333 0.8571428571428571)) '
                '(: rule_2 (Implication (Premises (audience-expertise $_14360 "intermediate")) (Conclusions (engagement $_14360 "Very_High"))) (STV 0.7777777777777778 0.9722222222222222)))'
            ]
        )


if __name__ == "__main__":
    unittest.main()
