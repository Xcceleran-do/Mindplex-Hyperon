;;;;;;;;  The implementation of the Rule Learning over Knowledge Graphs with Genetic Logic Programming paper using metta ;;;;;;;;

;; a function to count the examples that satisfy all the given atoms in the list
    ;; Args:
        ;; $rule: list containing a head and body atoms and
        ;; $kb: The knowledge base to search from
    ;; Returns: the number of examples that satisfy all the atoms in $rule as a float
(: countSupport (-> Expression $space Number))
(= (countSupport $rule $kb)
    (function 
        (chain (eval (cons-atom , $rule)) $rule'
        (chain (eval (collapse (match $kb $rule' $rule))) $supList
        (chain (eval (size-atom $supList)) $sup
        (chain (eval (* $sup 1.0)) $floatSup
        (return $floatSup)))))))
        
;; a function to calculate the standard confidence and used as a fitness function for the evoda
    ;; Args:
        ;; $rule: list containing a head and body atoms and
        ;; $kg: The knowledge base to search from
    ;; Returns: the standard confidence of a given rule $rule over a knowledge graph $kg
(: stdConfidence (-> Expression $space Number))
(= (stdConfidence $rule $kg)
    (if-decons-expr $rule $head $body
    (let* (
        ($bodySupport (countSupport $body $kg))
        ($ruleSupport (countSupport $rule $kg))
        ($isByZero (== $bodySupport 0))
        ($r (if $isByZero 0 (/ $ruleSupport $bodySupport)))
        ($bool (isnan-math $r))
        ($res (if $bool 0 $r))
    )
    $res
    )
    0)
)
;; a function to select top 2 fittest rules from the population 
    ;; Args: 
        ;; $pop: The population of rules
        ;; $fitnessFn: The fitness function
        ;; $kg: The knowledge graph
    ;; Returns: Expression containing the top 2 fittest rules
(: select (-> Expression (-> Expression Expression Number) $space Expression))
(= (select $pop $fitnessFn $kg) 
    (let* (
        ($fitnesses (map-atom $pop $ind ($fitnessFn $ind $kg)))
        ($totalFitness (sum $fitnesses))
        ($probabilities (map-atom $fitnesses $f (if (== $totalFitness 0) 0 (/ $f $totalFitness))))
        ($zipped (zip $pop $probabilities))
        ($sorted (selectionSort $zipped 2 bySecondAtom ()))
        ($top (map-atom $sorted $pair (first-from-pair $pair)))
        ($takes (take 2 $top))
    )
    $takes))

;; a function to gets all the predicates from the given knowledge graph
(: getPredicates (-> Atom Expression))
(= (getPredicates $kg)
    (function 
    (chain (eval (collapse (match $kg ($predicate' ($subjType $subject) ($objType $object)) ($predicate' $subjType $objType)))) $predicates
    (chain (eval (unique-atom $predicates)) $uniquePredicates
    (return $uniquePredicates)))))

;; A function to mutate a single atom. It generates 3 random floats and compare each with the mutationRate to decide wheather to mutate the predecate or/and varialbes or leave it as it is.
(: singleMutate (-> Expression Expression Atom Number Expression Expression))
(= (singleMutate $pop ($predicate ($subjType $subject) ($objType $object)) $kg $mutationRate $variables)
    (let*(
        ($rand1 (random-float &rng 0 1))
        ($bool (< $rand1 $mutationRate))
        ($predicates (getPredicates $kg))
        ((($randPred $sType $oType) $predIdx) (randomAtom $predicates))
        ($rand2 (random-float &rng 0 1))
        ($bool2 (< $rand2 $mutationRate))
        ($pop' (flatten $pop))
        (($randSub $sIdx) (randomAtom $variables))
        ($rand3 (random-float &rng 0 1))
        ($bool3 (< $rand2 $mutationRate))
        (($randObj $oIdx) (randomAtom $variables))
    )  
    (let ($mutatePred ($sT $s) ($oT $o)) (if $bool ($randPred ($sType $subject) ($oType $object)) ($predicate ($subjType $subject) ($objType $object)))
        (let ($p $mutateSub $o') (if $bool2 ($mutatePred ($sT $randSub) ($oT $o)) ($mutatePred ($sT $s) ($oT $o)))
            (if $bool3 ($p $mutateSub ($oT $randObj)) ($p $mutateSub $o'))))
    ))

;; A function that may add a new atom to a rule based on the mutationRate and also call singleMutate on every atom of the rule
(: mutate (-> Expression Expression Number Atom Expression Expression))
(= (mutate $rule $pop $mutationRate $kg $variables)
    (let*(
        ($rand (random-float &rng 0 1))
        ($bool (< $rand $mutationRate))
        ($newAtom (if $bool (genAtom $kg $variables) ()))
        ($isUnique (not (isMember $newAtom $rule)))
        ($rule' (if (and $isUnique $bool) (appendAtom $rule $newAtom) $rule))
    )
    (map-atom $rule' $atom (eval (singleMutate $pop $atom $kg $mutationRate $variables))))) 


;; A function to generate a random atom
(: genAtom (-> Atom Expression Expression))
(= (genAtom $kg $variables)
    (let* (
        ($predicates (getPredicates $kg))
        ((($randPred $subType $objType) $predIdx) (randomAtom $predicates))
        (($randSub $subIdx) (randomAtom $variables))
        (($randObj $objIdx) (randomAtom $variables))
    )
    ($randPred ($subType $randSub) ($objType $randObj))
    )
    )


;; a function to perform the crossover. It takes two rules, randomly chose one from each and swap them. It returns two new rules
(: crossover (-> Expression Expression Number Expression))
(= (crossover $rule1 $rule2 $crossoverRate)
    (let $random (random-float &rng 0 1)
    (if (> $random $crossoverRate)
    ($rule1 $rule2)
    (let*(
        (($head1 $body1) (decons-atom $rule1))
        (($head2 $body2) (decons-atom $rule2))
        (($atom1 $idx1) (randomAtom $body1))
        (($atom2 $idx2) (randomAtom $body2))
        (($left1 $right1) ((take $idx1 $body1) (drop (+ $idx1 1) $body1)))
        (($left2 $right2) ((take $idx2 $body2) (drop (+ $idx2 1) $body2)))
        ($crossBody1 (union-atom $left1 ($atom2)))
        ($crossBody2 (union-atom $left2 ($atom1)))
        ($crossRule1 (cons-atom $head1 $crossBody1))
        ($crossRule2 (cons-atom $head2 $crossBody2))
    )
    ((union-atom $crossRule1 $right1) (union-atom $crossRule2 $right2))
    ))))

;; a function which manage the whole proccess of learning a rule. It generate initial population based on the target using singlton copy rule with a size of $popSize. Gets the argmax based on the ftness function and generate population by calling generatePop function.
(: ruleLearning (-> Atom Expression Number Number (-> Expression Atom Number) Number Number Expression Expression))
(= (ruleLearning $kg $target $popSize $maxGen $fitnessFn $mutationRate $crossoverRate $variables)
    (let*(
        ($initialPop (initialPop $target $popSize $kg $variables))
        ($bestSet ((argmax $initialPop $fitnessFn $kg)))
        ($genList (range 1 $maxGen))
        ($newBest (foldl-atom $genList $initialPop $accum $genNum (generatePop $bestSet $accum $kg $target $popSize $maxGen $fitnessFn $mutationRate $crossoverRate $variables)))
        (($first $second) (select $newBest $fitnessFn $kg))
    )
    $first
    )
    )

;; A function to generate a population of $popSize by selecting two parents based on the select function and make a crossover and mutation to generate a new rule. Evaluate the new rule if it's a novel and better one. If so it adds it to bestSet. It runs until the bestSet size reaches $popSize
(: generatePop (-> Expression Expression Atom Expression Number Number (-> Expression Atom Number) Number Number Expression Expression))
(= (generatePop $bestSet $initialPop $kg $target $popSize $maxGen $fitnessFn $mutationRate $crossoverRate $variables)
    (let*(
        (($parent1 $parent2) (select $initialPop $fitnessFn $kg))
        (($child1 $child2) (crossover $parent1 $parent2 $crossoverRate))
        ($child1' (mutate $child1 $initialPop $mutationRate $kg $variables))
        ($child2' (mutate $child2 $initialPop $mutationRate $kg $variables))
        ($newBest (let $newBest' (if (isValidNewRule $child1' $bestSet $kg) (appendAtom $bestSet $child1') $bestSet) (if (isValidNewRule $child2' $newBest' $kg) (appendAtom $newBest' $child2') $newBest')))
        ($bestSize (size-atom $newBest))
    )
    (if (< $bestSize $popSize) (generatePop $newBest $initialPop $kg $target $popSize $maxGen $fitnessFn $mutationRate $crossoverRate $variables) $newBest
    )))

;; a function to cheak if a rule is valid to add it to bestSet. It checks if it is unique and have at least a support from the kg
(: isValidNewRule (-> Expression Expression Atom Bool))
(= (isValidNewRule $rule $bestSet $kg)
    (let*(
        ($isUnique (not (isMember $rule $bestSet)))
        ($support (countSupport $rule $kg))
        ($hasEnoughSupport (>= $support 1))
    )
    (and $isUnique $hasEnoughSupport)
    )
    )

;; a function to initialize a new population set using a singltone copy rule 
(: initialPop (-> Expression Number Atom Expression Expression))
(= (initialPop $target $popSize $kg $variables)
    (let*(
        ($predicates' (getPredicates $kg))
        ($atoms (consAtom $predicates' $popSize () $kg $target $variables))
    )
    $atoms
    ))

;; a helper function for initialPop to generate atoms with a singltone copy rule on the target atom.
(: consAtom (-> Expression Number Expression Atom Expression Expression Expression))
(= (consAtom $preds $popSize $acc $kg ($head ($sType $s)  ($oType $o)) $variables)
    (if (> $popSize 0)
        (let*(
            ((($pred $subjectType $objectType) $idx) (randomAtom $preds))
            (($newO $objIdx) (randomAtom $variables))
            ($bool (== $pred $head))
            ($count (countSupport (($pred ($subjectType $s) ($objectType $o))) $kg))
            ($bool2 (< $count 1))
            (($accum $popSize') (if (or $bool2 $bool) ($acc $popSize) ((cons-atom (($head ($sType $s)  ($oType $o)) ($pred ($subjectType $s) ($objectType $newO))) $acc) (- $popSize 1))))
            )
            (consAtom $preds $popSize' $accum $kg ($head ($sType $s)  ($oType $o)) $variables))
        $acc))

;; a function to pick the top one based on the fitness function
(= (argmax $expr $f $kg)
    (let*(
        ($values (map-atom $expr $arg (argHelper $f $arg $kg)))
        ($sorted (selectionSort $values 1 bySecondAtom ()))
        ($selected (car-atom $sorted))
        )
    (car-atom $selected)))

;; a helper function for argmax
(= (argHelper $f $arg $kg)
    (let $res ($f $arg $kg) ($arg $res)))

;; A master function to generate a required number of rules by callin the rulelearning function which generate a single rule
(= (ruleCovering $maxRules $kg $target $popSize $maxGen $fitnessFn $mutationRate $crossoverRate $variables $acc)
    (let*(
        ($curRule (ruleLearning $kg $target $popSize $maxGen $fitnessFn $mutationRate $crossoverRate $variables))
        ($fitness ($fitnessFn $curRule $kg))
        ($curRule' (cons-atom , $curRule))
        ($newAcc (cons-atom ($curRule $fitness) $acc))
        ($remaining (collapse (get-atoms $kg)))
        ($maxRules' (- $maxRules 1))
    )
    (if (or (> $maxRules' 0) (== $remaining ())) (ruleCovering $maxRules' $kg $target $popSize $maxGen $fitnessFn $mutationRate $crossoverRate $variables $newAcc) $newAcc)
    )
)