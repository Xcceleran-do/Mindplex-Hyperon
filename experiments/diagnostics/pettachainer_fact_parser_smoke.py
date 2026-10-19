from experiments.mining_api import parse_facts_for_pettachainer, select_facts_for_prompt


def main() -> None:
    sample = """[
    ((: (fact:- (audience-expertise A_14219 "advanced")) (audience-expertise A_14219 "advanced") (STV 0.9 0.95))
     (: (fact:- (title A_14219 "A title with (paren)")) (title A_14219 "A title with (paren)") (STV 1.0 1.0))
     (: fact10 (engagement A_14219 "Low") (STV 0.1 0.9)))
    ]"""

    facts = parse_facts_for_pettachainer(sample)
    assert len(facts) == 3, facts
    assert facts[0].startswith("(: (fact:- (audience-expertise")
    assert facts[1].startswith("(: (fact:- (title")
    assert facts[2].startswith("(: fact10")

    selected = select_facts_for_prompt(facts * 50, "(engagement A_14219 $x)", limit=2)
    assert len(selected) == 2, selected
    assert any("(engagement A_14219" in item for item in selected), selected

    print("PeTTaChainer fact parser smoke test passed.")


if __name__ == "__main__":
    main()
