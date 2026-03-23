 ;; Rule to calculate the empirical truth value of a pattern.

 ;;;;;;;;;;
 ;; Type ;;
 ;;;;;;;;;;


 ;; =============================================================================
;; Type: TruthValue
;; -----------------------------------------------------------------------------
;; Purpose: Represents a truth value with confidence measurement
;; Constructor: (EMPTV <strength> <confidence>)
;; =============================================================================
; (: TruthValue Type)
; (: EMPTV (-> $a $b TruthValue))



;; =============================================================================
;; Constant: Default_k
;; -----------------------------------------------------------------------------
;; Purpose:
;;   Provides a default smoothing constant used in confidence calculations.
;;
;; Value:
;;   - 1
;;
;; Usage:
;;   Used in the denominator of the count_to_confidence function to avoid
;;   overconfidence with small sample sizes.
;; =============================================================================
(= (Default_k) 1)


;; =============================================================================
;; Function: count_to_confidence
;; -----------------------------------------------------------------------------
;; Purpose: 
;;   Calculate a confidence value based on a count using a smoothing constant.
;;
;; Parameters:
;;   - $x: The count value (e.g., number of observations or samples)
;;
;; Internal Calculation:
;;   1. Adds the smoothing constant Default_k to the input count
;;   2. Divides the count by the sum to compute a normalized confidence
;;      => confidence = x / (x + Default_k)
;;
;; Returns:
;;   - A confidence value between 0 and 1
;; =============================================================================
(= (count_to_confidence $x) (// $x (+ $x (Default_k))))


;; =============================================================================
;; Function: emp-tv
;; -----------------------------------------------------------------------------
;; Purpose: Calculate empirical truth value for a pattern in database
;; Parameters:
;;   - $pattern: Target pattern to evaluate
;;   - $db: Database to analyze
;; 
;; Internal Computation:
;;   1. universe-count: Get total relevant items
;;   2. sup-num: Get pattern occurrences
;;   3. confidence: Calculate using count_to_confidence
;;   4. mean: Calculate support ratio
;; 
;; Returns:
;;   - EMPTV structure containing mean and confidence

(= (emp-tv $pattern $db)
    (let*
        (  
             ($_ (println! "======= Emperical Truth Value ======="))
            ($antecedent (antecedent $pattern))
            ( $sup-ante (sup-num $db $antecedent))
            ( $sup (sup-num $db $pattern))
            ( $conf (count_to_confidence $sup))   
            ($mean (// $sup $sup-ante))
        )
    (STV $mean  $conf)))