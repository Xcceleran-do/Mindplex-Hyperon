from __future__ import annotations

import unittest
from unittest.mock import Mock

from experiments.services.nl2pln_client import NL2PLNClient, NL2PLNError


class TestNL2PLNClient(unittest.TestCase):
    def build_client(self, response: Mock) -> tuple[NL2PLNClient, Mock]:
        session = Mock()
        session.headers = {}
        session.post.return_value = response
        client = NL2PLNClient(
            base_url="http://nl2pln:8080",
            api_key="s" * 32,
            session=session,
        )
        return client, session

    def test_translates_with_predicate_allowlist_and_context(self) -> None:
        response = Mock(ok=True)
        response.json.return_value = {
            "queries": [
                {
                    "source": '(: $proof (tone A_1 "Analytical") $tv)',
                    "source_query_index": 0,
                }
            ]
        }
        client, session = self.build_client(response)

        query = client.translate_query(
            "What tone does A_1 have?",
            [
                '(: fact_1 (tone A_1 "Analytical") (STV 1 1))',
                '(: rule_1 (Implication (Premises (tone $x "Analytical")) (Conclusions (engagement $x "High"))) (STV 1 1))',
            ],
        )

        self.assertEqual(query, '(: $proof (tone A_1 "Analytical") $tv)')
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["context"]["predicates"], ["engagement", "tone"])
        self.assertEqual(session.headers["Authorization"], f"Bearer {'s' * 32}")

    def test_rejects_missing_compiled_query(self) -> None:
        response = Mock(ok=True)
        response.json.return_value = {"queries": []}
        client, _session = self.build_client(response)

        with self.assertRaises(NL2PLNError):
            client.translate_query(
                "What tone does A_1 have?",
                ['(: fact_1 (tone A_1 "Analytical") (STV 1 1))'],
            )


if __name__ == "__main__":
    unittest.main()
