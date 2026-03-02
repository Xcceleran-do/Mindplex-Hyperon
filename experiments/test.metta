!(import! &self /mnt/c/Users/HP/iCog/Mindplex-Hyperon/PeTTa/lib/lib_import.metta)
!(import_prolog_functions_from_file "/mnt/c/Users/HP/iCog/Mindplex-Hyperon/experiments/frequent-pattern-miner/conj_exp.pl" (unique_combinations_star cut-first-char sort_conj))

!(import! &self /mnt/c/Users/HP/iCog/Mindplex-Hyperon/experiments/frequent-pattern-miner/frequent-pattern-miner)
!(import! &self /mnt/c/Users/HP/iCog/Mindplex-Hyperon/experiments/pattern-miner/pattern-miner)
!(import! &self /mnt/c/Users/HP/iCog/Mindplex-Hyperon/experiments/utils/common-utils)
!(import! &tempo /mnt/c/Users/HP/iCog/Mindplex-Hyperon/experiments/atomspace_visualizer/public/data)
!(import! &self /mnt/c/Users/HP/iCog/Mindplex-Hyperon/experiments/chainer/main)

!(let $atom (let $fact (get-atoms &tempo) (: (fact:- $fact) $fact)) (add-atom &res1 $atom))

!(let $atom (get-atoms &tempo) (add-atom &purifiedDbSpace $atom))


!(let $pat (pattern-miner &purifiedDbSpace 3 3) (let $atom (main $pat) (add-atom &res1 $atom)) )
; !(get-atoms &res1)
!(backward-chain &res1 (S (S Z)) (: $prf (engagement A_16624 "Low")))
