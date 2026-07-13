from __future__ import annotations

import unittest

from experiments.api.chainer import select_facts_for_query


class TestChainerQueryFactSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.facts = [
            '(: (fact:- (tone A_1 "informative")) (tone A_1 "informative") (STV 1 1))',
            '(: (fact:- (engagement A_1 "Low")) (engagement A_1 "Low") (STV 0.4 0.9))',
            '(: (fact:- (tone A_10 "critical")) (tone A_10 "critical") (STV 1 1))',
        ]

    def test_concrete_query_selects_only_the_named_article(self) -> None:
        selected = select_facts_for_query(
            self.facts,
            '(: $proof (engagement A_1 "Low") $tv)',
        )

        self.assertEqual(selected, self.facts[:2])

    def test_variable_query_keeps_the_complete_dataset(self) -> None:
        selected = select_facts_for_query(
            self.facts,
            '(: $proof (engagement $article "Low") $tv)',
        )

        self.assertEqual(selected, self.facts)

    def test_unknown_concrete_article_does_not_load_unrelated_facts(self) -> None:
        selected = select_facts_for_query(
            self.facts,
            '(: $proof (engagement A_999 "Low") $tv)',
        )

        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
