!(register-module! ../DFOL)
!(register-module! ../utilities)
!(import! &self utilities:helperFunctions)
!(import! &self DFOL:numpy:numme)

!(bind! dict ((1 X) (2 Y) (3 A) (4 B) (5 C) (6 D))) ;; I don't know if it is better to pass the dict as a parameter. I'll use it as a global 'var' for the time being. I think this can also be constracted from the input in side the function.

;; TODO: By scanning the dataset once, the number of binary predicates nb and the number of unary predicates nu can be determined

(= (propositionalization $X $Y $varList $d $targetAtom $featureSet $postiveExamples $traningFact)
    (let*(
        ($newList  (collapse (addToVarDom (superpose $varList) $traningFact)))
        (($newX $newY)  (foldl-atom $postiveExamples ($X $Y) $comulative $example (newSet $targetAtom   $example $comulative $traningFact)))
        ($allVars (cons-atom $newY $newList))
        ($init (collapse (let $x (superpose $newX) ($x))))
        ($allSubs (foldl-atom  $allVars $init $acc $cur (cartesianProd $cur $acc)))
        ($featSize (size-atom $featureSet))
        (() (println! $allSubs))
        ($trainableSet (map-atom $allSubs $curSub (genInterpretationPair $curSub $featureSet $postiveExamples $traningFact (np.zeros $featSize) (np.zeros 1) $targetAtom)))
        (() (println! $trainableSet))
        )
        (refineDS $trainableSet)
        ))

(= (addToVarDom $var $traningFact)
        (let $collapsed (collapse ((let ($r $s $o) (superpose $traningFact) (superpose ($s $o))) $var))
            (unique-atom $collapsed))
    )


(= (newSet ($pred $sub $obj) $example $comulative $traningFact)
    (unify ($pred $newSub $newObj) $example
        (let ($X $Y) $comulative 
            ((cons-atom ($newSub 1) $X) (cons-atom ($newObj 2) $Y)))
            $comulative)
    )

(= (newSet ($pred $sub) $example $comulative $traningFact)
    (unify ($pred $newSub) $example
        (let* ((($X $Y) $comulative)
                )
            ((cons-atom ($newSub 1) $X) (addToVarDom 2 $traningFact)))
            $comulative)
    )

(= (genInterpretationPair $curSub $featureSet $postiveExamples $traningFact $vi $vo $targetAtom)
    (let*(
        ($pairs ($vi $vo 0))
        (($finalVi $finalVo $idx) (foldl-atom $featureSet $pairs $updated $feat (substitution $feat $traningFact $postiveExamples $updated $curSub $targetAtom)))
    )
    ($finalVi $finalVo)
    )
    )

(= (substitution $feat $traningFact $postiveExamples ($vi $vo $idx) $curSub $targetAtom)
    (let*(
            ($` (if (isMember (ground $feat $curSub) $traningFact) 
                    (np.put $vi $idx 1) $vi)) 
            ($_ (if (isMember (ground $targetAtom $curSub) $postiveExamples) (np.put $vo 0 1) $vo))
            ($nxtIdx (+ $idx 1))
            )
            ($vi $vo $nxtIdx) ;; since np.put update the array in place I returned $vi itself
            ))

(= (ground ($pred $sub $obj) $curSub)
    (let*(
        ($groundedSub (getSub (getId $sub) $curSub))
        ($groundedObj (getSub (getId $obj) $curSub))
    )
    ($pred $groundedSub $groundedObj)
    )
    )
    

(= (ground ($pred $sub) $curSub)
    (let*(
        ($groundedSub (getSub (getId $sub) $curSub)) 
    )
    ($pred $groundedSub)
    )
    )

(= (getId $sub) 
 (let $sup (superpose dict) (unify $sup ($id $sub) $id Empty)))

(= (getSub $id $curSub)
    (let ($sub $id) (superpose $curSub) $sub))

(= (refineDS $trainableSet)
    (let $all (map-atom $trainableSet $cur (rmZeros $cur)) (removeNonDet $all ()))
    )

(= (rmZeros $cur)
    (let* (
            ($vi (car-atom $cur))
            ($sum (np.sum $vi))
            ) 
            (if-equal $sum 0 () $cur)
            ))

; !(propositionalization
;     ()
;     ()
;     () ;; Assuming varList are X and Y
;     2      ;; An example depth value
;     (cyclic X)
;     ((edge X Y) (path X Y))
;     ((cyclic a) (cyclic b))
;     ((edge a b) (path a b) (edge b c) (path a d))
; )

; !(match &self (, (cyclic $a) (edge $a $b) (path $a $b)) ($a $b))



; Excellent! Moving on to the implementation of the neural network architecture described in Section 3.2 of the sources is a great next step.

; Based on the detailed descriptions in the source material, here is a full Python code implementation for the neural network portion of the DFOL model. This code assumes you have already performed the propositionalization step (Section 3.1) and have the interpretation vectors `vi` (input) and `vo` (output) ready for training.

; The implementation uses the PyTorch library, as it is well-suited for building neural networks with custom loss functions and gradient-based optimization, which is central to the DFOL model.

; ### Python Implementation of DFOL's Neural Network (Section 3.2)

; ```python
; import torch
; import torch.nn as nn
; import torch.optim as optim
; import torch.nn.functional as F
; import math

; # Based on the sources, information outside of this code (like the exact
; # value of hyperparameters) would be required to run it. This implementation
; # uses PyTorch, a common machine learning library. You may need to install it
; # (`pip install torch`) and independently verify its usage for your specific needs.

; class DFOL_Network(nn.Module):
;     """
;     A PyTorch implementation of the DFOL neural network architecture
;     as described in Section 3.2 of the source paper.
;     """
;     def __init__(self, C, m1, m2, na, V_size, t, gamma=1.0):
;         """
;         Initializes the trainable tensors and hyperparameters.
        
;         Args:
;             C (int): Number of valid body features (|vi|).
;             m1 (int): Hyperparameter for the number of logic rules in MS_P.
;             m2 (int): Hyperparameter for the number of rules headed by the target atom (in MA_P).
;             na (int): Hyperparameter for the number of rules headed by auxiliary predicates.
;             V_size (int): Total number of unique variables, |V|.
;             t (int): Arity of the head atom.
;             gamma (float): Hyperparameter for the slope of the sigmoid-like activation function φ.
;         """
;         super(DFOL_Network, self).__init__()

;         # --- Trainable Tensors ---
;         # MS_P: Encodes the primary set of logic rules.
;         self.MS_P = nn.Parameter(torch.rand(m1, C))
        
;         # MA_P: Encodes rules for the target and auxiliary predicates.
;         self.MA_P = nn.Parameter(torch.rand(m2, na, C))
        
;         # --- Store Hyperparameters ---
;         self.C = C
;         self.m = m1 + m2  # Total number of rules
;         self.m1 = m1
;         self.m2 = m2
;         self.na = na
;         self.t = t
;         self.V_size = V_size
;         self.gamma = gamma

;     def _activation_phi(self, x):
;         """ Differentiable sigmoid-like activation function φ(x) = 1 / (1 + e^(-γx)). """
;         return torch.sigmoid(self.gamma * x)

;     def _fuzzy_or(self, x, dim=0):
;         """ Differentiable fuzzy OR operation using product t-norm: ∨̃_i x_i = 1 - ∏_i(1 - x_i). """
;         return 1 - torch.prod(1 - x, dim=dim)

;     def _fuzzy_and(self, x, dim=0):
;         """ Differentiable fuzzy AND operation using product t-norm: ∧̃_i x_i = ∏_i x_i. """
;         return torch.prod(x, dim=dim)

;     def forward(self, vi):
;         """
;         Performs the forward pass of the network to compute the predicted output v̄o.
        
;         Args:
;             vi (torch.Tensor): A batch of input interpretation vectors. Shape: (batch_size, C).
        
;         Returns:
;             torch.Tensor: The predicted output vector v̄o. Shape: (batch_size, 1).
;             torch.Tensor: The final SH matrix MP.
;         """
;         # --- Step 1: Merge and Concatenate to form MP ---
;         # Merge MA_P along the 'na' dimension.
;         MA_prime_P = (1 / self.na) * torch.sum(self.MA_P, dim=1) # Shape: (m2, C)
        
;         # Concatenate MS_P and MA'_P vertically to form the final SH matrix MP.
;         # MP has shape (m, C), where m = m1 + m2.
;         MP = torch.cat((self.MS_P, MA_prime_P), dim=0)

;         # --- Step 2: Differentiable Inference ---
;         # MP[k,·] x v_i^T for each rule k and each input vector in the batch.
;         # Shape: (batch_size, m)
;         inference_scores = torch.matmul(vi, MP.T)
        
;         # Apply activation function φ(score - 1).
;         activated_scores = self._activation_phi(inference_scores - 1)
        
;         # Apply fuzzy OR across all rules to get the final prediction v̄o.
;         # Shape: (batch_size, 1)
;         v_o_bar = self._fuzzy_or(activated_scores, dim=1).unsqueeze(1)
        
;         return v_o_bar, MP

;     def compute_losses(self, MP, vo, v_o_bar, basic_embeddings, occ_embeddings, M_prior=None):
;         """
;         Computes all the loss components described in the paper.
        
;         Args:
;             MP (torch.Tensor): The final SH matrix from the forward pass.
;             vo (torch.Tensor): The ground truth output interpretation vectors.
;             v_o_bar (torch.Tensor): The predicted output interpretation vectors.
;             basic_embeddings (torch.Tensor): b(α) for all C features. Shape: (C, t).
;             occ_embeddings (torch.Tensor): o(α) for all C features. Shape: (C, |V|-t).
;             M_prior (torch.Tensor, optional): Matrix of previously learned sound rules for curriculum learning.
        
;         Returns:
;             dict: A dictionary containing all computed loss values.
;         """
        
;         # --- LI: Inference Loss ---
;         # Binary Cross-Entropy between predicted and true output.
;         L_I = F.binary_cross_entropy(v_o_bar, vo)

;         # --- LS: Sum Loss ---
;         # Enforces that the sum of each row in MP is equal to 1.
;         row_sums = torch.sum(MP, dim=1)
;         L_S = F.mse_loss(row_sums, torch.ones_like(row_sums))

;         # --- LB: Basic Loss ---
;         # Enforces that all head variables appear in the rule body.
;         # This requires iterating through each of the 'm' rules.
;         basic_k_scores = []
;         for k in range(self.m):
;             # MP[k, i] * b(α_i) for all i features.
;             possibilistic_b_embeddings = MP[k, :].unsqueeze(1) * basic_embeddings
;             # This is M_k^b, with shape (C, t).
            
;             # Fuzzy OR down each of the 't' columns.
;             # "Does head variable 'j' appear in *any* body atom?"
;             var_present_scores = self._fuzzy_or(possibilistic_b_embeddings, dim=0)
            
;             # Fuzzy AND across the 't' results.
;             # "Do *all* head variables appear?"
;             basic_k = self._fuzzy_and(var_present_scores)
;             basic_k_scores.append(basic_k)
        
;         basic_k_tensor = torch.stack(basic_k_scores)
;         L_B = F.mse_loss(basic_k_tensor, torch.ones_like(basic_k_tensor))

;         # --- LO: Occurrence Loss ---
;         # Penalizes variables not in the head from appearing only once.
;         F_values = []
;         # Hyperparameters for the measurement function F(x).
;         a, b, c, d = 1.0, 1.0, 10.0, 1.0
        
;         for k in range(self.m):
;             # MP[k, i] * o(α_i) for all i features.
;             possibilistic_o_embeddings = MP[k, :].unsqueeze(1) * occ_embeddings
;             # This is M_k^o, with shape (C, |V|-t).
            
;             # V_k^o: Sum of occurrences for each "other" variable.
;             V_k_o = torch.sum(possibilistic_o_embeddings, dim=0)
            
;             # F(x) = a * exp(b - c(x - d)^2).
;             # This penalizes values close to d=1.
;             F_k = a * torch.exp(b - c * (V_k_o - d)**2)
;             F_values.append(torch.sum(F_k)) # Sum penalty over all "other" variables.

;         L_O = torch.mean(torch.stack(F_values))

;         # --- LF: Auxiliary Predicate Format Loss ---
;         # Encourages diversity in rules for auxiliary predicates by minimizing cosine similarity.
;         L_F_sum = 0.0
;         if self.na > 1: # Only if there are multiple auxiliary rules to compare.
;             for k in range(self.m2):
;                 # Iterate over all 2-combinations of rows in MA_P[k,:,:].
;                 for i1 in range(self.na):
;                     for i2 in range(i1 + 1, self.na):
;                         row1 = self.MA_P[k, i1, :]
;                         row2 = self.MA_P[k, i2, :]
;                         cos_sim = F.cosine_similarity(row1, row2, dim=0)
;                         L_F_sum += F.mse_loss(cos_sim, -torch.ones_like(cos_sim))
;         L_F = L_F_sum

;         # --- LC: Curriculum Learning Loss ---
;         # Encourages new rules to be different from previously learned sound rules.
;         L_C_sum = 0.0
;         if M_prior is not None:
;             for k1 in range(self.m):
;                 for k2 in range(M_prior.shape):
;                     cos_sim = F.cosine_similarity(MP[k1, :], M_prior[k2, :], dim=0)
;                     L_C_sum += F.mse_loss(cos_sim, -torch.ones_like(cos_sim))
;         L_C = L_C_sum
        
;         return {
;             "LI": L_I, "LS": L_S, "LB": L_B, 
;             "LO": L_O, "LF": L_F, "LC": L_C
;         }

; # --- Example Usage ---
; def train_dfol(vi_data, vo_data, C, m1, m2, na, V_size, t, b_embeddings, o_embeddings, epochs=1000):
    
;     # Instantiate the model
;     model = DFOL_Network(C=C, m1=m1, m2=m2, na=na, V_size=V_size, t=t)
    
;     # Hyperparameter vector for weighting losses
;     # These would be tuned based on the specific task.
;     theta = {
;         "LI": 1.0, "LS": 0.1, "LB": 0.5, 
;         "LO": 0.5, "LF": 0.1, "LC": 0.1
;     }

;     # Use the Adam optimizer
;     optimizer = optim.Adam(model.parameters(), lr=0.01)

;     # Convert data to tensors
;     vi_tensor = torch.tensor(vi_data, dtype=torch.float32)
;     vo_tensor = torch.tensor(vo_data, dtype=torch.float32)

;     print("--- Starting Training ---")
;     for epoch in range(epochs):
;         model.train()
        
;         # Zero the gradients
;         optimizer.zero_grad()
        
;         # Forward pass
;         v_o_bar, MP = model(vi_tensor)
        
;         # Compute all loss components
;         # M_prior would be updated every few epochs based on extracted sound rules
;         losses = model.compute_losses(MP, vo_tensor, v_o_bar, b_embeddings, o_embeddings, M_prior=None)
        
;         # Calculate the final weighted loss
;         final_loss = sum(theta[key] * losses[key] for key in losses if losses[key] != 0.0)
        
;         # Backward pass and optimization
;         final_loss.backward()
;         optimizer.step()
        
;         if epoch % 100 == 0:
;             print(f"Epoch {epoch}/{epochs}, Final Loss: {final_loss.item():.4f}")
;             print(f"  Losses -> LI: {losses['LI']:.4f}, LS: {losses['LS']:.4f}, LB: {losses['LB']:.4f}, LO: {losses['LO']:.4f}")
    
;     print("--- Training Finished ---")
    
;     # Return the trained matrix MP for rule extraction (Section 3.3)
;     _, final_MP = model(vi_tensor)
;     return final_MP.detach()

; if __name__ == '__main__':
;     # --- Dummy Data for Demonstration ---
;     # This setup simulates the grandparent task from the paper.
;     # grandparent(X,Y) <- ...
;     # Variables V = {X, Y, Z}, Head variables Vh = {X, Y}, Other variables Vo = {Z}
    
;     # Parameters for the Grandparent example
;     C = 13            # Number of valid features (e.g., mother(X,Z), father(Z,Y), etc.)
;     m1 = 4            # Let's assume 4 rules to learn
;     m2 = 2            # Rules with potential auxiliary predicates
;     na = 2            # Number of auxiliary rules
;     V_size = 3        # |V| = |{X, Y, Z}|
;     t = 2             # Arity of grandparent(X,Y) is 2

;     # --- Sample Embeddings (These must be pre-computed) ---
;     # This is an example based on Example 5
;     # Let's say feature 0 is mother(X,Z) -> b=, o=
;     # Let's say feature 1 is father(Z,Y) -> b=, o=
;     # ... and so on for all C=13 features.
    
;     # Dummy basic and occurrence embeddings.
;     # In a real scenario, these are determined by your feature set PF.
;     b_embeddings = torch.randint(0, 2, (C, t), dtype=torch.float32)
;     o_embeddings = torch.randint(0, 2, (C, V_size - t), dtype=torch.float32)

;     # --- Sample Interpretation Pairs (from Section 3.1) ---
;     # Assuming you have N data points from propositionalization.
;     # Each vi is a C-dimensional vector, each vo is a 1-dimensional vector.
;     N = 50 # Number of training examples (substitutions)
;     vi_data = torch.randint(0, 2, (N, C)).float()
;     vo_data = torch.randint(0, 2, (N, 1)).float()

;     # Run training
;     trained_MP = train_dfol(vi_data, vo_data, C, m1, m2, na, V_size, t, b_embeddings, o_embeddings)
    
;     print("\n--- Trained SH Matrix (MP) ---")
;     print(trained_MP)
;     print("\nThis matrix can now be used for rule extraction as per Section 3.3.")

; ```

; ### Explanation of the Code and its Connection to the Sources

; This code directly translates the mathematical formulas and concepts from Section 3.2 into a working PyTorch model.

; 1.  **Initialization (`__init__`)**:
;     *   It sets up the two main trainable tensors: `MS_P` and `MA_P`. These are initialized with random values, which the Adam optimizer will adjust during training.
;     *   It stores key hyperparameters like `C` (number of features), `m1`, `m2`, `na` (rule counts), `V_size`, and `t` (arity), which are needed for the loss calculations.

; 2.  **Forward Pass (`forward`)**:
;     *   **Matrix Merging**: It first performs the `merge` operation on `MA_P` by summing along the `na` dimension and then the `concat` operation to combine `MS_P` and the merged matrix, creating the final SH matrix `MP`.
;     *   **Differentiable Inference**: It computes the matrix product of the input `vi` and `MP`'s transpose. This is followed by the `φ(x - 1)` activation function and the fuzzy OR (`∨̃`) operation to produce the final prediction, `v̄o`. This sequence is a differentiable replacement for the immediate consequence operator.

; 3.  **Loss Functions (`compute_losses`)**:
;     *   **`LI` (Inference Loss)**: Uses Binary Cross-Entropy to measure the difference between the model's prediction `v̄o` and the true label `vo`.
;     *   **`LS` (Sum Loss)**: A simple Mean Squared Error loss that pushes the sum of values in each row of `MP` towards 1.
;     *   **`LB` (Basic Loss)**: This is the most complex part. It builds the `Mk b` matrix for each rule by multiplying the rule's weights (`MP[k,:]`) with the pre-computed `basic_embeddings`. It then applies the fuzzy OR (`∨̃`) down the columns and a fuzzy AND (`∧̃`) across the results to get the `basick` score, which is then compared to 1 using MSE.
;     *   **`LO` (Occurrence Loss)**: Similar to `LB`, it computes `Mk o` using the `occurrence_embeddings`. It then sums the columns to get `Vko` and applies the measurement function `F(x)` to penalize variables that occur only once.
;     *   **`LF` and `LC` (Cosine Similarity Losses)**: These losses use `F.cosine_similarity` to encourage diversity. `LF` pushes the auxiliary rule embeddings in `MA_P` to be dissimilar (similarity of -1), while `LC` does the same for the main matrix `MP` relative to previously learned `M_prior` rules.

; 4.  **Training Loop (`train_dfol`)**:
;     *   The main function orchestrates the training process. In each epoch, it performs a forward pass, computes the weighted sum of all losses, and uses the Adam optimizer to update the model's parameters (`MS_P` and `MA_P`) based on the computed gradients.
;     *   The `theta` dictionary holds the weights for each loss component, allowing you to control the influence of each constraint.

; After training, the `trained_MP` tensor contains the learned weights, representing the discovered logic program. You would then apply the rule extraction methods described in Section 3.3 to this matrix to get the final, human-readable symbolic rules.