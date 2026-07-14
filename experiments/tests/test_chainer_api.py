from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.api import chainer
from experiments.api.chainer import select_facts_for_query


class TestChainerQueryFactSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.facts = [
            '(: (fact:- (tone A_1 "informative")) (tone A_1 "informative") (STV 1 1))',
            '(: (fact:- (engagement A_1 "Low")) (engagement A_1 "Low") (STV 0.4 0.9))',
            '(: (fact:- (tone A_10 "critical")) (tone A_10 "critical") (STV 1 1))',
        ]

    def test_concrete_engagement_query_excludes_the_target_label(self) -> None:
        selected = select_facts_for_query(
            self.facts,
            '(: $proof (engagement A_1 "Low") $tv)',
        )

        self.assertEqual(selected, [self.facts[0]])

    def test_other_queries_keep_engagement_as_a_possible_input(self) -> None:
        selected = select_facts_for_query(self.facts, '(tone A_1 "informative")')

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

    def test_query_worker_uploads_only_selected_facts(self) -> None:
        uploaded = []

        class FakeClient:
            def backward(self, _kb_id, _query, _depth):
                return ["proof"]

        def ensure_remote(facts):
            uploaded.extend(facts)
            return FakeClient(), "kb"

        with patch.object(chainer, "reload_petta_dataset_if_ready"), \
             patch.object(chainer, "dataset_facts", return_value=self.facts), \
             patch.object(chainer, "ordered_chainer_rules", return_value=[]), \
             patch.object(chainer, "ensure_remote_chainer", side_effect=ensure_remote):
            proofs = chainer.run_chainer_query_worker('(engagement A_1 "Low")', depth=3)

        self.assertEqual(uploaded, [self.facts[0]])
        self.assertEqual(proofs, ["proof"])


if __name__ == "__main__":
    unittest.main()
