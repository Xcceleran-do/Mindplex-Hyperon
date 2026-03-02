;; STV Test File - Separate from main.metta
;; Purpose: Test STV integration without modifying main code

;; Import required modules
!(import! &self experiments:chainer:main)
!(import! &self experiments:chainer:facts)
!(import! &self experiments:chainer:rules)
!(import! &stv-formulas experiments:PLN:Formulas)

;; Initialize STV knowledge base using existing atomspace
!(add-reduct &res1 (get-atoms &facts))
!(add-reduct &res1 (get-atoms &rules))

;; Test 1: Basic STV backward chaining
!(println "=== Test 1: Basic STV Backward Chaining ===")
!(backward-chain &res1 (fromNumber 3) (: $prf (engagement 9 "high")))

;; Test 2: STV with different article
!(println "=== Test 2: STV with Different Article ===")
!(backward-chain &res1 (fromNumber 3) (: $prf (engagement 8 "high")))

;; Test 3: STV with medium engagement
!(println "=== Test 3: STV with Medium Engagement ===")
!(backward-chain &res1 (fromNumber 3) (: $prf (engagement 4 "medium")))

;; Test 4: Check STV knowledge base contents
!(println "=== Test 4: STV Knowledge Base Contents ===")
!(get-atoms &res1)

;; Test 5: Compare with regular backward chaining
!(println "=== Test 5: Compare with Regular ===")
!(println "STV Result:")
!(backward-chain &res1 (fromNumber 2) (: $prf (engagement 9 "high")))

!(println "Legacy Result:")
!(backward-chain &res1 (fromNumber 2) (: $prf (engagement 1 "high")))

!(println "=== STV Testing Complete ===")
