(= (propositionalization $X $Y $varList $targetAtom $featureSet $postiveExamples $traningFact)
    (let*(
        ($newList  (collapse (addToVarDom (superpose $varList) $traningFact)))
        (($newX $newY)  (foldl-atom $postiveExamples ($X $Y) $comulative $example 
                        (updateTargetVar $targetAtom $example $comulative $traningFact)))
        ($allVars (cons-atom $newY $newList))
        ($init (cons-atom $newX $allVars))
        ($allSubs (multiCartesian $init))
        ($uniqueAll (unique-atom $allSubs))
        ($featSize (size-atom $featureSet))
        ($trainableSet (foldl-atom $uniqueAll (() ()) $acc $curSub 
                            (genInterpretationPair $curSub $featureSet $postiveExamples $traningFact 
                                (repeat 0.0 $featSize) 0.0 $targetAtom $acc)))
        )
        $trainableSet
        ))

(= (addToVarDom $var $traningFact)
        (let $collapsed (collapse ((let ($r $s $o) (superpose $traningFact) (superpose ($s $o))) $var))
            (unique-atom $collapsed))
    )


(= (updateTargetVar $atom $example $comulative $traningFact)
    (unify ($pred $sub $obj) $atom 
        (let*(
            (($pred $newSub $newObj) $example)
            (($X $Y) $comulative)
            ) 
            ((cons-atom ($newSub 1) $X) (cons-atom ($newObj 2) $Y))
            )
        (let*(
            (($pred $newSub) $example)
            (($X $Y) $comulative)
            )
            ((cons-atom ($newSub 1) $X) (addToVarDom 2 $traningFact))) ; 1 refers to X and 2 refers to Y in the ((1 X) (2 Y) (7 Z) (3 A) (4 B) (5 C) (6 D)). TODO: avode the use of majic numbers
            )
    )


(= (genInterpretationPair $curSub $featureSet $postiveExamples $traningFact $vi $vo $targetAtom ($setVi $setVo))
    (let*(
        (($finalVi $finalVo $idx) (foldl-atom $featureSet ($vi $vo 0) $updated $feat 
                                        (substitution $feat $traningFact $postiveExamples $updated $curSub $targetAtom)))
        )
    (if (> (sum $finalVi) 0) ((cons-atom $finalVi $setVi) (cons-atom $finalVo $setVo)) ($setVi $setVo))
    )
    )
    
(= (substitution $feat $traningFact $postiveExamples ($vi $vo $idx) $curSub $targetAtom)
    (let*(
            ($featGrd (ground $feat $curSub))
            ($updatedVi (if (isMember $featGrd $traningFact) 
                    (replaceByIndex $vi $idx 1.0) $vi)) 
            ($tarGrd (ground $targetAtom $curSub))
            ($updatedVo (if (isMember $tarGrd $postiveExamples) (1.0) $vo))
            ($nxtIdx (+ $idx 1))
            )
            ($updatedVi $updatedVo $nxtIdx) 
            ))

(= (ground $atom $curSub)
    (unify $atom ($pred $sub $obj)
        (let*(
            ($groundedSub (getSub (getId $sub) $curSub))
            ($groundedObj (getSub (getId $obj) $curSub))
        )
        ($pred $groundedSub $groundedObj)
        )
        (let*(
            (($unaryPred $unarySub) $atom)
            ($groundedSub (getSub (getId $sub) $curSub)) 
        )
        ($unaryPred $groundedSub)
        )
    )
    )
    
    
(= (getId $sub) 
    (let $sup (superpose ((1 X) (2 Y) (7 Z) (3 A) (4 B) (5 C) (6 D))) (unify $sup ($n $sub) $n (empty))))

(= (getSub $id $curSub)
    (let ($sub $id) (superpose $curSub) $sub))


(= (basicEmbedding $head $body)
    (unify $body ($pred $sub) 
        (let*(
            (($headPred $x) $head)
            ($subId (getId $sub))
            ($isSub (if (== (getId $x) $subId) 1 0))
            )
            $isSub
        )
        (let*(
            (($binaryPred $binarySub $binaryObj) $body)
            (($headPred $x $y) $head)
            ($subId (getId $binarySub))
            ($objId (getId $binaryObj))
            ($isSub (if (or (== (getId $x) $subId) (== (getId $y) $subId)) 1 0))
            ($isObj (if (or (== (getId $y) $objId) (== (getId $x) $objId)) 1 0))
        )
        ($isSub $isObj)
    )
))

(= (occurrenceEmbedding $noneHeadVars $body)
    (unify $body ($pred $sub $obj)
        (map-atom $noneHeadVars $cur (if (isMember $cur ($sub $obj)) 1 0))
        (let ($unaryPred $unarySub) $body 
            (map-atom $noneHeadVars $cur (if (== $cur $unarySub) 1 0))
        )
        )
)