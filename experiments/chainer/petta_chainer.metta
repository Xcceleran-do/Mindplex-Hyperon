!(import! &self compile)

;;;;;;;;;
;; Nat ;;
;;;;;;;;;

;; Define Nat
(: Nat Type)
(: Z Nat)
(: S (-> Nat Nat))

;; Define cast functions between Nat and Number
(: fromNumber (-> Number Nat))
(= (fromNumber $n) (if (<= $n 0) Z (S (fromNumber (- $n 1)))))
(: fromNat (-> Nat Number))
(= (fromNat Z) 0)
(= (fromNat (S $k)) (+ 1 (fromNat $k)))

;;;;;;;;;;;;;
;; Chainer ;;
;;;;;;;;;;;;;

(= (chainer $depth $goal) (match &kb $goal $goal))

(= (chainer $depth (CPU $fun $args $res)) (let $call (cons $fun $args) (let $res (eval $call) (CPU $fun $args $res))))

(= (chainer (S $depth) $goal)
  (let $_ (match &kb (rules ($args |- $goal)) (rules ($args |- $goal)))
  (let $args (chain-rec $depth $args)
  $goal)))

(= (chain-rec $depth ($goal Nil))  ((chainer $depth $goal) Nil))
(= (chain-rec $depth ($goal $dep)) ((chainer $depth $goal) (chain-rec $depth $dep)))

;; Simple probabilistic/fuzzy formulas used by the compiled chainer.
;; Strength for implication application is damped by antecedent strength.
;; Confidence must also depend on both the rule and the antecedent; otherwise
;; uncertain facts can produce fully rule-confident conclusions.
(= (MpFormula (STV $abs $abc) (STV $as $ac)) (STV (* $abs $as) (* $abc $ac)))
(= (AndFormula (STV $si $ci) (STV $sa $ca)) (let $sb (min $si $sa)
                                              (let $cb (min $ci $ca) (STV $sb $cb))))

(= (OrFormula (STV $si $ci) (STV $sa $ca)) (let $sb (max $si $sa)
                                             (let $cb (min $ci $ca) (STV $sb $cb))))

(= (LikelierThanFormula (STV $sa $ca) (STV $sb $cb)) (let $c (min $ca $cb) (if (> $sa $sb) (STV $sa $c) (STV $sb $c))))


(= (AndProjection (STV $s $c)) (STV $s $c))
(= (OrProjection (STV $s $c)) (STV $s $c))

(= (NotFormula (STV $s $c) (STV (- 1 $s) $c)))

(= (clamp01 $x) (min 1.0 (max 0.0 $x)))

;; Given A, B, and A -> B, estimate B -> A with Bayes' rule:
;; P(A|B) = P(B|A) * P(A) / P(B)
(= (InversionFormula (STV $sa $ca) (STV $sb $cb) (STV $sab $cab))
   (let $sba (if (<= $sb 0.0) 0.0 (clamp01 (/ (* $sab $sa) $sb)))
        (STV $sba (min-atom ($ca $cb $cab)))))

(= (Equal $a $b) (== $a $b))
(= (NotEqual $a $b) (not (== $a $b)))


(= (compileadd $kb $stmt)
  (let $atoms (mm2compile $kb $stmt) (add-atom &kb $atoms)))


(= (query $depth $kb $stmt)
  (let (cons (goal $query) $adds) (collapse (mm2compileQuery $kb $stmt))
  (let $_ (if (= $adds ()) $_ (let $sadds (superpose $adds) (add-atom &kb $sadds)))
       (pretty (chainer $depth $query)))))

!(add-atom &kb (rules (((CPU $f $args $res) Nil) |- (: $kb cpu (Compute $f $args $res) (STV 1.0 1.0)))))
