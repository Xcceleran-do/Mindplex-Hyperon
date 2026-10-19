!(import! &self experiments/frequent-pattern-miner/conj_exp)

;; Placeholder strings stay inert while combinations are built, then become
;; real, consistently shared variables at the output boundary.
!(assertEqual
   (get-metatype (parse "$x"))
   Variable)

!(assertEqual
   (clause-placeholders (author "$x" "$y"))
   ("$x" "$y"))

;; Four engagement pairs are valid. The author/follows pair is excluded because
;; it shares both $x and $y rather than exactly one hub.
!(assertEqual
   (size-atom
      (unique-combinations-star
         ((topic "$x" AI)
          (engagement "$x" low)
          (tone "$x" calm)
          (author "$x" "$y")
          (follows "$x" "$y"))
         2))
   4)
