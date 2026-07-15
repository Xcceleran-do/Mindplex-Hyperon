from __future__ import annotations

import unittest

from experiments.api.chat.support import (
    article_exists_in_facts,
    deterministic_engagement_query,
    handle_backward_chain_for_message,
)


class TestChatBackwardChainGuard(unittest.TestCase):
    def test_very_high_engagement_keeps_canonical_bucket_name(self) -> None:
        for level in ("very high", "very-high", "very_high"):
            with self.subTest(level=level):
                query = deterministic_engagement_query(
                    f"Why does A_24867 have {level} engagement?",
                    [],
                )

                self.assertEqual(
                    query,
                    '(: $prf (engagement A_24867 "Very_High") $tv)',
                )

    def test_article_presence_uses_exact_atom_token(self) -> None:
        facts = ['(: fact_1 (tone A_142190 "informative") (STV 1 1))']

        self.assertFalse(article_exists_in_facts("A_14219", facts))
        self.assertTrue(article_exists_in_facts("A_142190", facts))

    def test_missing_article_skips_backward_chainer(self) -> None:
        chainer_called = False

        def get_chainer_result(_query: str):
            nonlocal chainer_called
            chainer_called = True
            return {}

        response, calls = handle_backward_chain_for_message(
            "Why does article A_14219 have low engagement?",
            get_all_facts_and_rules=lambda: {
                "status": "success",
                "facts": ['(: fact_1 (tone A_24014 "informative") (STV 1 1))'],
            },
            translate_query=lambda _message, _facts: "(: $proof (engagement A_14219 \"Low\") $tv)",
            get_chainer_result=get_chainer_result,
            logger=None,
        )

        self.assertFalse(chainer_called)
        self.assertIn("not in the active knowledge base", response)
        self.assertEqual(calls[-1]["name"], "check_article_in_knowledge_base")

    def test_nontrivial_question_uses_nl2pln_translation(self) -> None:
        translated_queries = []

        response, calls = handle_backward_chain_for_message(
            "What tone can be proved for article A_1?",
            get_all_facts_and_rules=lambda: {
                "status": "success",
                "facts": ['(: fact_1 (tone A_1 "Analytical") (STV 1 1))'],
            },
            translate_query=lambda message, facts: translated_queries.append((message, facts))
            or '(: $proof (tone A_1 "Analytical") $tv)',
            get_chainer_result=lambda query: {
                "status": "success",
                "justification": query,
            },
            logger=None,
        )

        self.assertEqual(len(translated_queries), 1)
        self.assertIn('(tone A_1 "Analytical")', response)
        self.assertEqual(calls[0]["name"], "translate_query_nl2pln")


if __name__ == "__main__":
    unittest.main()
