import requests
import json
import os
import re


class MWJClient:
    """
    Drop-in replacement for PeTTa.
    The MWJ server is stateful — &kb persists between requests.
    No timing output is printed; behaviour matches PeTTa exactly.
    """

    _STRATEGY_DIRECT  = "direct"
    _STRATEGY_WRAPPED = "wrapped"
    _STRATEGY_SCAN    = "scan"

    def __init__(self, url="http://localhost:5001/metta"):
        self.url = url
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "text/plain; charset=utf-8"})
        self._strategy_cache: dict[str, str] = {}
        

    # ------------------------------------------------------------------
    # Low-level HTTP  (no printing)
    # ------------------------------------------------------------------

    def _post(self, body: str) -> list:
        response = self._session.post(
            self.url,
            data=body.encode("utf-8"),
            timeout=120,
        )
        response.raise_for_status()
        return self._parse_response(response.text)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> list:
        text = text.strip()
        if not text or text == "[]":
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [
                    item if isinstance(item, str) else str(item).lower()
                    for item in parsed
                ]
        except (json.JSONDecodeError, ValueError):
            pass

        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if inner:
                items = self._split_metta_list(inner)
                return [item.strip() for item in items if item.strip()]
            return []

        if "\n" in text:
            return [line.strip() for line in text.splitlines() if line.strip()]

        return [text]

    def _split_metta_list(self, text: str) -> list:
        items, current, depth, in_string = [], [], 0, False
        for char in text:
            if char == '"' and not in_string:
                in_string = True;  current.append(char)
            elif char == '"' and in_string:
                in_string = False; current.append(char)
            elif in_string:
                current.append(char)
            elif char == "(":
                depth += 1; current.append(char)
            elif char == ")":
                depth -= 1; current.append(char)
            elif char == "," and depth == 0:
                items.append("".join(current).strip()); current = []
            else:
                current.append(char)
        if current:
            items.append("".join(current).strip())
        return items

    # ------------------------------------------------------------------
    # KB lifecycle
    # ------------------------------------------------------------------

    def clear_kb(self, kb_id: str) -> None:
        """
        Remove only atoms belonging to kb_id from &kb.

        Safe for multi-client use — only atoms whose first argument matches
        kb_id are removed, so other clients' data on the shared server is
        never touched.

        Call this from PeTTaChainer.__init__ after generating self.kb:
            if USE_MWJ:
                self.handler.clear_kb(self.kb)
        """
        try:
            # Fetch only atoms that belong to this kb_id
            atoms = self._post(
                f'!(match &kb (: {kb_id} $prf $type $tv) '
                f'(: {kb_id} $prf $type $tv))'
            )
            for atom in atoms:
                self._session.post(
                    self.url,
                    data=f"!(remove-atom &kb {atom})".encode("utf-8"),
                    timeout=30,
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API  (drop-in for PeTTa)
    # ------------------------------------------------------------------

    def process_metta_string(self, query: str) -> list:
        query = query.strip()
        if re.search(r'!\(match\s+&kb\s+\(:\s+\S+\s+\$', query):
            results = self._handle_get_facts(query)
        else:
            results = self._post(query)
        return self._deduplicate(results)

    def load_metta_file(self, file_path: str) -> list:
        file_path = os.path.normpath(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            metta_code = f.read()
        return self.process_metta_string(metta_code)

    # ------------------------------------------------------------------
    # get_facts  —  strategy discovery cached per kb_id
    # ------------------------------------------------------------------

    def _handle_get_facts(self, query: str) -> list:
        m = re.match(
            r'!\(match\s+&kb\s+'
            r'\(:\s+(\S+)\s+(\$\w+)\s+(\$\w+)\s+(\$\w+)\)'
            r'\s+'
            r'\(:\s+\S+\s+\$\w+\s+\$\w+\s+\$\w+\)\)',
            query.strip(),
        )
        if not m:
            return self._post(query)

        kb_id = m.group(1)
        prf   = m.group(2)
        typ   = m.group(3)
        tv    = m.group(4)

        strategy = self._strategy_cache.get(kb_id)

        if strategy is None:
            if self._post(
                f'!(match &kb (: {kb_id} {prf} {typ} {tv}) '
                f'(: {kb_id} {prf} {typ} {tv}))'
            ):
                strategy = self._STRATEGY_DIRECT
            elif self._post(
                f'!(match &kb (mm2compile {kb_id} (: {prf} {typ} {tv})) '
                f'(: {kb_id} {prf} {typ} {tv}))'
            ):
                strategy = self._STRATEGY_WRAPPED
            else:
                strategy = self._STRATEGY_SCAN
            self._strategy_cache[kb_id] = strategy

        if strategy == self._STRATEGY_DIRECT:
            return self._post(
                f'!(match &kb (: {kb_id} {prf} {typ} {tv}) '
                f'(: {kb_id} {prf} {typ} {tv}))'
            )

        if strategy == self._STRATEGY_WRAPPED:
            return self._post(
                f'!(match &kb (mm2compile {kb_id} (: {prf} {typ} {tv})) '
                f'(: {kb_id} {prf} {typ} {tv}))'
            )

        # STRATEGY_SCAN
        all_atoms = self._post('!(match &kb $x $x)')
        return [a for a in all_atoms if a.startswith(f"(: {kb_id} ")]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(items: list) -> list:
        seen, seen_set = [], set()
        for item in items:
            if item not in seen_set:
                seen_set.add(item)
                seen.append(item)
        return seen
