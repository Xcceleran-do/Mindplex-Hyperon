!(import! &self ../PeTTa/lib/lib_import.metta)
!(import_prolog_functions_from_file "../experiments/frequent-pattern-miner/conj_exp.pl" (unique_combinations_star cut-first-char sort_conj))

!(import! &self ../experiments/frequent-pattern-miner/frequent-pattern-miner)
!(import! &self ../experiments/pattern-miner/pattern-miner)
!(import! &self ../experiments/utils/common-utils)
!(import! &tempo ../experiments/atomspace_visualizer/public/data)
!(import! &self ../experiments/chainer/main)

!(let $atom (let $fact (get-atoms &tempo) (: (fact:- $fact) $fact)) (add-atom &res1 $atom))

!(let $atom (get-atoms &tempo) (add-atom &purifiedDbSpace $atom))


!(let $pat (pattern-miner &purifiedDbSpace 3 3) (let $atom (main $pat) (add-atom &res1 $atom)) )
; !(get-atoms &res1)
!(backward-chain &res1 (S (S Z)) (: $prf (engagement A_16624 "Low")))
