!(register-module! ../../../Mindplex-Hyperon)
!(import! &self Mindplex-Hyperon:Utilities:helperFunctions)
!(import! &data Mindplex-Hyperon:Evoda:data:book-data)
!(import! &self Mindplex-Hyperon:Evoda:evoda)
!(bind! &kb (new-space))
!(add-reducts &kb (collapse (get-atoms &data)))
!(bind! rule1 ((has_read (User $x) (Book $y)) (has_read (User $z) (Book $y)) (friend (User $x) (User $z))));; user $x reads book $y because his friend $z has read the same book
!(bind! rule2 ((has_read (User $x) (Book $y)) (interested_in (User $x) (Genre $z)) (genre_of (Book $y) (Genre $z))));; user $x reads book $y with genre $z because he is interested in genre $z
!(bind! rule3 ((friend (User $x) (User $y)) (similar_interest (User $x) (User $y))))
;; Test cases for countSupport function 
!(assertEqual (countSupport rule1 &kb) 6) 
!(assertEqual (countSupport rule2 &kb) 25) 
!(assertEqual (countSupport rule3 &kb) 2) 

;; Test cases for stdConfidence
!(assertEqual (stdConfidence rule1 &kb) 0.8571428571428571)
!(assertEqual (stdConfidence rule2 &kb) 0.8620689655172413) 
!(assertEqual (stdConfidence rule3 &kb) 1) 

;; Test cases for select
!(assertEqual (select (rule1 rule2 rule3) stdConfidence &kb) (rule3 rule2))
!(assertEqual (select (rule1 rule2) stdConfidence &kb) (rule2 rule1))
!(assertEqual (select (rule1) stdConfidence &kb) (rule1))

; Test cases for getPredicates
!(assertEqual (getPredicates &kb) ((author_of Book Author) (genre_of Book Genre) (publication_year_of Book Year) (has_read User Book) (interested_in User Genre) (friend User User) (similar_interest User User) (similar_genre Book Book) (author_genre Author Genre) (favorite_author User Author) (language Book Language))) ; this may sometimes cause an error as it is non-deterministic

; ;; Test cases for singleMutate
!(assertEqual (let $mutated (singleMutate (rule1 rule2 rule3) (friend (User $x) (User $y)) &kb 1 ($a $b $c)) (isMember $mutated rule3)) False) ; I make the mutationRate 1 to make sure there is a mutation
!(assertEqual (let $unmutated (singleMutate (rule1 rule2 rule3) (friend (User $x) (User $y)) &kb 0 ($a $b $c)) (isMember $unmutated rule3)) True) ; I make the mutationRate 0 to make sure there isn't a mutation

;; Test cases for mutate
!(assertEqual (let $mutated (mutate rule1 (rule1 rule2 rule3) 1 &kb ($a $b $c)) (== rule1 $x)) False)
!(assertEqual (let $unmutated (mutate rule1 (rule1 rule2 rule3) 0 &kb ($a $b $c)) (== $unmutated rule1)) True)

;; Test case for genAtom
!(assertEqual (let $atom (genAtom &kb ($a $b $c)) (unify $atom ($randPred ($subType $randSub) ($objType $randObj)) True False)) True)

;; Test cases for crossover
!(assertEqual (let ($child1 $child2) (crossover rule1 rule2 1) (and (== rule1 $child1) (== rule2 $child2))) False)
!(assertEqual (let ($child1 $child2) (crossover rule1 rule2 0) (and (== rule1 $child1) (== rule2 $child2))) True)

;; Test cases for ruleLearning 
; !(assertEqual (let $rule (ruleLearning &kb (has_read (User "Saron Melaku") (Book $x)) 100 100 stdConfidence 0.1 0.7 ($a $b $c)) (> (countSupport $rule) 1)) True) ; we'll need to run it for significant amout of popSize and maxGen to gen a result which is not possible with metta
; !(assertEqual (let $rule (ruleLearning &kb (has_read (User $y) (Book $x)) 100 100 stdConfidence 0.1 $crossoverRate ($a $b $c)) (> (countSupport $rule) 1)) True) ; we'll need to run it for significant amout of popSize and maxGen to gen a result which is not possible with metta
; !(assertEqual (let $rule (ruleLearning &kb (has_read (User $y) (Book "Emegua")) 100 100 stdConfidence 0.1 $crossoverRate ($a $b $c)) (> (countSupport $rule) 1)) True) ; we'll need to run it for significant amout of popSize and maxGen to gen a result which is not possible with metta

;; Test cases for generatePop
; !(assertEqual (let $rules (generatePop ((freind (User "Saron Melaku") (User $y)) (similar_interest (User "Saron Melaku") (User $y))) &kb (freind (User "Saron Melaku") (User $y)) 100 100 stdConfidence 0.1 0.7 ($a $b $c)) (> (size-atom $rules) 100)) True) ; we'll need to run it for significant amout of popSize and maxGen to gen a result which is not possible with metta
; !(assertEqual (let $rules (generatePop ((freind (User $x) (User $y)) (similar_interest (User $x) (User $y))) &kb (freind (User $x) (user $y)) 100 100 stdConfidence 0.1 0.7 ($a $b $c)) (size-atom $rules)) 100) ; we'll need to run it for significant amout of popSize and maxGen to gen a result which is not possible with metta
; !(assertEqual (let $rules (generatePop ((freind (User $x) (User $y)) (similar_interest (User $x) (User $y))) &kb (freind (User $y) (user "Emegua")) 100 100 stdConfidence 0.1 0.7 ($a $b $c)) (size-atom $rules)) 100) ; we'll need to run it for significant amout of popSize and maxGen to gen a result which is not possible with metta

;; Test cases for isValidNewRule
!(assertEqual (isValidNewRule rule1 (rule2 rule3) &kb) True)
!(assertEqual (isValidNewRule rule1 (rule1 rule2 rule3) &kb) False)
!(assertEqual (isValidNewRule ((has_read (User $x) (Book $y)) (has_read (User $y) (Book $x))) (rule2 rule3) &kb) False) ; Impossible unless the book is named after the user :)

;; Test cases for initial population
!(assertEqual (let $pop (initialPop (has_read (User $x) (Book $y)) 5 &kb) (size-atom $pop)) 5)
!(assertEqual (let $pop (initialPop (has_read (User $x) (Book $y)) 2 &kb) (size-atom $pop)) 2) 

;; Test cases for consAtom
!(assertEqual (let $pop (consAtom ((author_of Book Author) (genre_of Book Genre) (publication_year_of Book Year) (has_read User Book) (interested_in User Genre) (friend User User) (similar_interest User User) (similar_genre Book Book) (author_genre Author Genre) (favorite_author User Author) (language Book Language)) 5 () &kb (has_read (User $x) (Book $y))) (size-atom $pop)) 5)
!(assertEqual (let $pop (consAtom ((author_of Book Author) (genre_of Book Genre) (publication_year_of Book Year) (has_read User Book) (interested_in User Genre) (friend User User) (similar_interest User User) (similar_genre Book Book) (author_genre Author Genre) (favorite_author User Author) (language Book Language)) 2 () &kb (has_read (User $x) (Book $y))) (size-atom $pop)) 2) 

;; Test cases for argmax
!(assertEqual (argmax (rule1 rule2 rule3) stdConfidence &kb) rule3)
!(assertEqual (argmax (rule1 rule2 rule3) countSupport &kb) rule2)

;; Test cases for ruleCovering 
!(assertEqual (let $rules (ruleCovering 5 &kb (has_read (User "Saron Melaku") (Book $x)) 2 1 stdConfidence 0.1 0.7 ($a $b $c) ()) (size-atom $rules)) 5) ; we'll need to run it for significant amout of popSize and maxGen to gen a result which is not possible with metta. This may fail if the &kb data is covered with less than 5 rules (rare case)
; !(assertEqual (let $rules  (ruleCovering 5 &kb (has_read (User $y) (Book $x)) 100 100 stdConfidence 0.1 0.7 ($a $b $c)) (size-atom $rules)) 5) ; we'll need to run it for significant amout of popSize and maxGen to gen a result which is not possible with metta. This may fail if the &kb data is covered with less than 5 rules (rare case)
; !(assertEqual (let $rules  (ruleCovering 5 &kb (has_read (User $y) (Book "Emegua")) 100 100 stdConfidence 0.1 0.7 5 ($a $b $c)) (size-atom $rules)) 5) ; we'll need to run it for significant amout of popSize and maxGen to gen a result which is not possible with metta. This may fail if the &kb data is covered with less than 5 rules (rare case)