from __future__ import annotations

from experiments.mining_api import create_app, get_chainer_service


def main() -> int:
    create_app()
    service = get_chainer_service()

    queries = [
        '(length A_16624 "Medium")',
        '(engagement A_16624 "Low")',
        '(content-type A_16624 "Tutorial")',
    ]
    for query in queries:
        proofs = service.query(query, depth=6)
        print(f"QUERY {query} COUNT {len(proofs)}")
        for proof in proofs:
            print(proof)
        print("---")

    service.add_forward_only_rule(
        '(: rule_existing_1 '
        '(-> (And (tone $x "Instructional") (reading-time $x "Medium")) '
        '(engagement $x "Low")) '
        '(STV 0.62 0.81))'
    )
    service.add_forward_only_rule(
        '(: rule_existing_2 '
        '(-> (And (tone $x "Instructional") (primary-goal $x "Inform")) '
        '(engagement $x "Low")) '
        '(STV 0.7 0.8))'
    )

    proofs = service.query('(: $prf (engagement A_16624 "Low") $tv)', depth=6)
    print(f"MIXED QUERY COUNT {len(proofs)}")
    for proof in proofs:
        print(proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
