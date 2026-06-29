"""
Unit tests for MWJClient — no live MWJ Docker container required.
Tests cover _parse_response, _split_metta_list, _handle_get_facts,
and clear_kb using unittest.mock to avoid real HTTP calls.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from experiments.services.mwj_client import MWJClient


# ------------------------------------------------------------------
# Helpers — create a client without hitting the network
# ------------------------------------------------------------------

def make_client() -> MWJClient:
    """Create MWJClient without triggering any HTTP call."""
    with patch("requests.Session"):
        client = MWJClient(url="http://localhost:5001/metta")
    return client


# ------------------------------------------------------------------
# _parse_response
# ------------------------------------------------------------------

class TestParseResponse:

    def test_empty_string_returns_empty_list(self):
        client = make_client()
        assert client._parse_response("") == []

    def test_empty_brackets_returns_empty_list(self):
        client = make_client()
        assert client._parse_response("[]") == []

    def test_whitespace_only_returns_empty_list(self):
        client = make_client()
        assert client._parse_response("   ") == []

    def test_json_string_array(self):
        client = make_client()
        assert client._parse_response('["a", "b", "c"]') == ["a", "b", "c"]

    def test_json_bool_array_lowercased(self):
        client = make_client()
        assert client._parse_response('[true, false]') == ["true", "false"]

    def test_metta_list_single_atom(self):
        client = make_client()
        result = client._parse_response('[(: kb1 fact1 (STV 0.9 0.9))]')
        assert result == ['(: kb1 fact1 (STV 0.9 0.9))']

    def test_metta_list_two_atoms(self):
        client = make_client()
        result = client._parse_response(
            '[(: kb1 fact1 (STV 0.9 0.9)),(: kb1 fact2 (STV 0.8 0.8))]'
        )
        assert len(result) == 2
        assert result[0] == '(: kb1 fact1 (STV 0.9 0.9))'
        assert result[1] == '(: kb1 fact2 (STV 0.8 0.8))'

    def test_newline_separated_values(self):
        client = make_client()
        result = client._parse_response("line1\nline2\nline3")
        assert result == ["line1", "line2", "line3"]

    def test_single_value(self):
        client = make_client()
        assert client._parse_response("hello") == ["hello"]

    def test_strips_whitespace_from_single_value(self):
        client = make_client()
        assert client._parse_response("  hello  ") == ["hello"]


# ------------------------------------------------------------------
# _split_metta_list
# ------------------------------------------------------------------

class TestSplitMettaList:

    def test_single_atom(self):
        client = make_client()
        result = client._split_metta_list('(: kb1 fact1 (STV 0.9 0.9))')
        assert result == ['(: kb1 fact1 (STV 0.9 0.9))']

    def test_two_atoms(self):
        client = make_client()
        result = client._split_metta_list(
            '(: kb1 fact1 (STV 0.9 0.9)),(: kb1 fact2 (STV 0.8 0.8))'
        )
        assert len(result) == 2
        assert result[0] == '(: kb1 fact1 (STV 0.9 0.9))'
        assert result[1] == '(: kb1 fact2 (STV 0.8 0.8))'

    def test_comma_inside_nested_parens_not_split(self):
        client = make_client()
        # The comma inside (STV 0.9, 0.9) should NOT split
        result = client._split_metta_list(
            '(: kb1 fact1 (length A "Medium") (STV 0.9 0.9)),'
            '(: kb1 fact2 (tone A "High") (STV 0.8 0.8))'
        )
        assert len(result) == 2

    def test_comma_inside_quoted_string_not_split(self):
        client = make_client()
        # The comma inside "Hi, there" should NOT split
        result = client._split_metta_list(
            '(: kb1 fact1 (tone A "Hi, there") (STV 0.9 0.9)),'
            '(: kb1 fact2 (length A "Medium") (STV 0.8 0.8))'
        )
        assert len(result) == 2
        assert '"Hi, there"' in result[0]

    def test_empty_string_returns_empty(self):
        client = make_client()
        result = client._split_metta_list('')
        assert result == []

    def test_deeply_nested_parens(self):
        client = make_client()
        result = client._split_metta_list(
            '(: kb1 fact1 (partial length (A "Medium")) (STV 0.9 0.9))'
        )
        assert len(result) == 1
        assert 'partial length' in result[0]


# ------------------------------------------------------------------
# _handle_get_facts  (mocked _post)
# ------------------------------------------------------------------

class TestHandleGetFacts:

    def test_unknown_query_shape_passes_through(self):
        client = make_client()
        client._post = MagicMock(return_value=["result"])
        result = client._handle_get_facts("!(some-other-query)")
        client._post.assert_called_once_with("!(some-other-query)")
        assert result == ["result"]

    def test_direct_strategy_cached_and_used(self):
        client = make_client()
        fake_facts = ['(: kb123 fact1 (STV 0.9 0.9))']
        client._post = MagicMock(return_value=fake_facts)

        query = '!(match &kb (: kb123 $prf $type $tv) (: kb123 $prf $type $tv))'
        result = client._handle_get_facts(query)

        assert result == fake_facts
        assert client._strategy_cache.get("kb123") == MWJClient._STRATEGY_DIRECT

    def test_strategy_cached_skips_probe_on_second_call(self):
        client = make_client()
        fake_facts = ['(: kb123 fact1 (STV 0.9 0.9))']
        client._post = MagicMock(return_value=fake_facts)
        client._strategy_cache["kb123"] = MWJClient._STRATEGY_DIRECT

        query = '!(match &kb (: kb123 $prf $type $tv) (: kb123 $prf $type $tv))'
        client._handle_get_facts(query)

        # Only one _post call (the final fetch), no probe
        assert client._post.call_count == 1

    def test_scan_strategy_filters_by_kb_id(self):
        client = make_client()
        client._strategy_cache["kb999"] = MWJClient._STRATEGY_SCAN
        all_atoms = [
            '(: kb999 fact1 (STV 0.9 0.9))',
            '(: kb000 fact1 (STV 0.5 0.5))',  # different kb_id — should be filtered out
            '(: kb999 fact2 (STV 0.8 0.8))',
        ]
        client._post = MagicMock(return_value=all_atoms)

        query = '!(match &kb (: kb999 $prf $type $tv) (: kb999 $prf $type $tv))'
        result = client._handle_get_facts(query)

        assert len(result) == 2
        assert all('kb999' in atom for atom in result)


# ------------------------------------------------------------------
# clear_kb  (mocked session)
# ------------------------------------------------------------------

class TestClearKb:

    def test_clear_kb_only_removes_matching_kb_id_atoms(self):
        client = make_client()
        kb_id = "kb123"
        matching_atoms = [
            f'(: {kb_id} fact1 (STV 0.9 0.9))',
            f'(: {kb_id} fact2 (STV 0.8 0.8))',
        ]
        # _post returns the atoms for this kb_id
        client._post = MagicMock(return_value=matching_atoms)
        client._session = MagicMock()

        client.clear_kb(kb_id)

        # Should post a scoped match query for kb_id
        client._post.assert_called_once()
        call_args = client._post.call_args[0][0]
        assert kb_id in call_args
        assert '$prf' in call_args

        # Should remove exactly the returned atoms
        assert client._session.post.call_count == len(matching_atoms)
        for call, atom in zip(client._session.post.call_args_list, matching_atoms):
            assert atom.encode("utf-8") in call[1].get("data", b"") or \
                   atom in call[1].get("data", b"").decode("utf-8")

    def test_clear_kb_does_not_crash_on_server_error(self):
        client = make_client()
        client._post = MagicMock(side_effect=Exception("server down"))
        # Should not raise
        client.clear_kb("kb123")


# ------------------------------------------------------------------
# _deduplicate
# ------------------------------------------------------------------

class TestDeduplicate:

    def test_removes_duplicates(self):
        result = MWJClient._deduplicate(["a", "b", "a", "c", "b"])
        assert result == ["a", "b", "c"]

    def test_preserves_insertion_order(self):
        result = MWJClient._deduplicate(["c", "a", "b", "a", "c"])
        assert result == ["c", "a", "b"]

    def test_empty_list(self):
        assert MWJClient._deduplicate([]) == []

    def test_no_duplicates_unchanged(self):
        items = ["x", "y", "z"]
        assert MWJClient._deduplicate(items) == items
