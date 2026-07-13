from __future__ import annotations

import unittest

from experiments.api.chat.support import article_exists_in_facts, handle_backward_chain_for_message


class TestChatBackwardChainGuard(unittest.TestCase):
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
            select_facts_for_prompt=lambda facts, _query, _limit: facts,
            call_asi_api=lambda _messages: {},
            system_instruction="test",
            get_chainer_result=get_chainer_result,
            logger=None,
        )

        self.assertFalse(chainer_called)
        self.assertIn("not in the active knowledge base", response)
        self.assertEqual(calls[-1]["name"], "check_article_in_knowledge_base")


if __name__ == "__main__":
    unittest.main()
