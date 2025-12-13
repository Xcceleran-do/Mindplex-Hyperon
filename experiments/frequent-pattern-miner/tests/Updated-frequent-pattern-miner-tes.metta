!(import! &self ../freq-pat)
!(import! &self ../../atomspace_visualizer/public:data)

;; lets transfer this data to other space
!(bind! dataBase (new-space))
!(let* 
    (
        ($x (get-atoms &self))
        ($debug1 (remove-atom &self $x))
        ($debug2 (add-atom dataBase $x))
    ) ())
; !(get-atoms dataBase)

!(import! &self ../Updated-frequent-pattern-miner)
!(import! &self ../Updated-conjunction-expansion)
!(import! &self ../Updated-frequentMinerHelperFunctions)

; !(import! &self ../use-python)
; !(get-atoms &self)

;; space for the final conjuncts
!(bind! conjunctSpace (new-space))

; !(frequency-pattern-miner dataBase-space conjunct-Space min-Support number-of-pattern-in-one-conjunct minus 2)
!(frequency-pattern-miner dataBase conjunctSpace 2 0)
