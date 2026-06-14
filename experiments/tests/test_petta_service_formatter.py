import unittest

from experiments.services.petta_service import PeTTaService


class TestPeTTaServiceFormatter(unittest.TestCase):
    def test_pattern_to_rule_keeps_lifted_variable(self) -> None:
        service = PeTTaService(project_root=".", setup_metta="")
        pattern = (
            'supportOf ((And (size_bucket $_123 "Low") (tone_bucket $_123 "Analytical") '
            '(engagement $_123 "High")) (STV 0.62 0.81)) 4'
        )

        rule = service.pattern_to_rule(pattern, 1)

        self.assertEqual(
            rule,
            '(: rule_1 (-> (And (size_bucket $x "Low") (tone_bucket $x "Analytical")) '
            '(engagement $x "High")) (STV 0.62 0.81))',
        )

    def test_formatter_inserts_single_lifted_rule_per_pattern(self) -> None:
        service = PeTTaService(project_root=".", setup_metta="")
        inserted = []
        service.add_forward_only_rule = lambda atom: inserted.append(atom) or ["true"]  # type: ignore[method-assign]

        result = service.formatter(
            {
                "patterns": [
                    {
                        "pattern": (
                            'supportOf ((And (size_bucket $_123 "Low") (tone_bucket $_123 "Analytical") '
                            '(engagement $_123 "High")) (STV 0.62 0.81)) 4'
                        )
                    }
                ]
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["insertedRuleCount"], 1)
        self.assertEqual(
            inserted,
            [
                '(: rule_1 (-> (And (size_bucket $x "Low") (tone_bucket $x "Analytical")) '
                '(engagement $x "High")) (STV 0.62 0.81))'
            ],
        )


if __name__ == "__main__":
    unittest.main()
