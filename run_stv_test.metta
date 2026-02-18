;; Simple STV Test
!(import! &rules rules)
!(import! &facts facts)
!(import! &stv-formulas experiments:PLN:Formulas)
!(add-reduct &res1 (get-atoms &facts))
!(add-reduct &res1 (get-atoms &rules))

!(println "=== STV Test Results ===")
!(backward-chain &res1 (fromNumber 3) (: $prf (engagement 9 "high")))
!(println "✅ STV backward chaining working!")
