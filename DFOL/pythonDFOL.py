import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class DFOL_Network(nn.Module):
    """
    A PyTorch implementation of the DFOL neural network architecture
    as described in Section 3.2 of the source paper.
    """
    def __init__(self, C, m1, m2, na, V_size, t, gamma=10.0):
        """
        Initializes the trainable tensors and hyperparameters.
        
        Args:
            C (int): Number of valid body features (|vi|).
            m1 (int): Hyperparameter for the number of logic rules in MS_P.
            m2 (int): Hyperparameter for the number of rules headed by the target atom (in MA_P).
            na (int): Hyperparameter for the number of rules headed by auxiliary predicates.
            V_size (int): Total number of unique variables, |V|.
            t (int): Arity of the head atom.
            gamma (float): Hyperparameter for the slope of the sigmoid-like activation function φ.
        """
        super(DFOL_Network, self).__init__()

        # --- Trainable Tensors ---
        # MS_P: Encodes the primary set of logic rules.
        # Initialize with very small random values to avoid saturation
        self.MS_P = nn.Parameter(torch.rand(m1, C) * 0.01 + 0.01)
        
        # MA_P: Encodes rules for the target and auxiliary predicates.
        # Initialize with very small random values to avoid saturation
        self.MA_P = nn.Parameter(torch.rand(m2, na, C) * 0.01 + 0.01)
        
        # --- Store Hyperparameters ---
        self.C = C
        self.m = m1 + m2  # Total number of rules
        self.m1 = m1
        self.m2 = m2
        self.na = na
        self.t = t
        self.V_size = V_size
        self.gamma = gamma

    def _activation_phi(self, x):
        """ Differentiable sigmoid-like activation function φ(x) = 1 / (1 + e^(-γx)). """
        return torch.sigmoid(self.gamma * x)

    def _fuzzy_or(self, x, dim=-1):
        """ Differentiable fuzzy OR operation using product t-norm: ∨̃_i x_i = 1 - ∏_i(1 - x_i). """
        return 1 - torch.prod(1 - x, dim=dim)

    def _fuzzy_and(self, x, dim=-1):
        """ Differentiable fuzzy AND operation using product t-norm: ∧̃_i x_i = ∏_i x_i. """
        return torch.prod(x, dim=dim)

    def forward(self, vi):
        """
        Performs the forward pass of the network to compute the predicted output v̄o.
        Following paper equation: v̄o = ∨_{k=1}^m φ(MP[k,·] × vi - 1)
        
        Args:
            vi (torch.Tensor): A batch of input interpretation vectors. Shape: (batch_size, C).
        
        Returns:
            torch.Tensor: The predicted output vector v̄o. Shape: (batch_size, 1).
            torch.Tensor: The final SH matrix MP.
        """
        
        # Merge operation: MA'_P = (1/na) * sum_i MA_P[:,i,:]  
        MA_prime_P = (1 / self.na) * torch.sum(self.MA_P, dim=1)
        
        # Concatenate: MP = concat(MS_P, MA'_P)
        MP = torch.cat((self.MS_P, MA_prime_P), dim=0)

        # Forward inference: MP[k,·] × vi for each rule k
        inference_scores = torch.matmul(vi, MP.T)  # (batch_size, m)
        
        # φ(MP[k,·] × vi - 1) - sigmoid activation with bias -1
        activated_scores = self._activation_phi(inference_scores - 1) 
        # ∨_{k=1}^m - fuzzy OR over all rules
        v_o_bar = self._fuzzy_or(activated_scores, dim=1).unsqueeze(1)
        
        return v_o_bar, MP

# --- Loss Function Class ---
class LossFunction():
    def __init__(self, m, m1, m2, na, fuzzy_or, fuzzy_and, MA_P):
        self.m = m
        self.m1 = m1
        self.m2 = m2
        self.na = na
        self.fuzzy_or = fuzzy_or
        self.fuzzy_and = fuzzy_and
        self.MA_P = MA_P

    def forward(self, MP, vo, v_o_bar, basic_embeddings, occ_embeddings,
                a, b, c, d, M_prior=None):
        # --- LI: Inference Loss ---
        # Ensure v_o_bar and vo have same shape
        # v_o_bar has shape (batch_size, 1), vo has shape (batch_size,)
        # We need to flatten v_o_bar to match vo shape for consistent loss calculation

        v_o_bar_flat = v_o_bar.squeeze(1) 
        L_I = F.binary_cross_entropy(v_o_bar_flat, vo)

        # --- LS: Row Sum Loss (encourages ~1.0 per row) ---
        row_sums = torch.sum(MP, dim=1)
        L_S = F.mse_loss(row_sums, torch.ones_like(row_sums))

        # --- LB: Basic Variable Coverage Loss (differentiable fuzzy ops) ---
        basic_losses = torch.tensor(0.0, device=MP.device)
        for k in range(self.m):
            weighted_embeddings = MP[k, :].unsqueeze(1) * basic_embeddings  
            var_coverage = self.fuzzy_or(weighted_embeddings, dim=0)  
            all_vars_covered = self.fuzzy_and(var_coverage, dim=0)
            basic_losses += F.mse_loss(all_vars_covered, torch.ones_like(all_vars_covered, device=all_vars_covered.device))

        L_B = basic_losses

        # --- LO: Occurrence Loss (with proper target comparison) ---
        F_values = []
        for k in range(self.m):
            V_k_o = torch.sum(MP[k, :].unsqueeze(1) * occ_embeddings, dim=0)
            F_k = a * torch.exp(b - c * (V_k_o - d)**2)
            F_values.append(torch.sum(F_k)) 
            
        L_O = torch.mean(torch.stack(F_values))

        # --- LF: Diversity Loss (auxiliary predicate format) ---
        L_F_sum = torch.tensor(0.0, device=MP.device)
        
        for k in range(self.m2):
            for i1 in range(self.na):
                for i2 in range(i1+1, self.na):
                    row1 = self.MA_P[k, i1, :]
                    row2 = self.MA_P[k, i2, :]
                    cos_sim = F.cosine_similarity(row1, row2, dim=0)
                    target = -torch.ones_like(cos_sim, device=cos_sim.device)
                    L_F_sum = L_F_sum + F.mse_loss(cos_sim, target)
        L_F = L_F_sum


        # --- LC: Curriculum Loss (optional) ---
        L_C_sum = 0.0
        if M_prior is not None:
            for k1 in range(self.m):
                for k2 in range(M_prior.shape[0]):
                    cos_sim = F.cosine_similarity(MP[k1, :], M_prior[k2, :], dim=0)
                    # encourage dissimilarity from prior
                    L_C_sum += F.mse_loss(cos_sim, - torch.ones_like(cos_sim))
        L_C = L_C_sum

        return {
            "LI": L_I, "LS": L_S, "LB": L_B,
            "LO": L_O, "LF": L_F, "LC": L_C
        }


# trainer class for DFOL_Network
class DFOLTrainer:
    def extract_rules(self, MP, feature_names, threshold=0.5):
        """
        Extract rules from the trained MP matrix using a threshold.
        Returns a list of (rule_index, [feature_names]) for each rule.
        """
        rules = []
        m, C = MP.shape
        for k in range(m):
            row = MP[k, :]
            active = (row > threshold).nonzero(as_tuple=True)[0]
            if len(active) > 0:
                body = [feature_names[i] for i in active.tolist()]
                rules.append((k, body))
        return rules
    def __init__(self, vi_tensor, vo_tensor, C, m1, m2, na, V_size, t, b_embeddings, o_embeddings, a, b, c, d, epochs=1000, lr=0.01, theta=None):
        self.model = DFOL_Network(C=C, m1=m1, m2=m2, na=na, V_size=V_size, t=t)
        self.epochs = epochs
        self.b_embeddings = b_embeddings
        self.o_embeddings = o_embeddings
        self.vi_tensor = vi_tensor
        self.vo_tensor = vo_tensor
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Balanced loss weights - strengthen LI (inference loss) much more
        self.theta = theta if theta else {"LI": 10.0, "LS": 15.5, "LB": 1.5, "LO": 1.5, "LF": 50.5, "LC": 1.1}
        self.M_prior = None
        # Instantiate the loss function class
        self.loss_fn = LossFunction(
            m= m1+m2, m1=m1, m2=m2, na=na,
            fuzzy_or=self.model._fuzzy_or,
            fuzzy_and=self.model._fuzzy_and,
            MA_P=self.model.MA_P
        )
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            v_o_bar, MP = self.model(self.vi_tensor)
                
            losses = self.loss_fn.forward(MP, self.vo_tensor, v_o_bar, self.b_embeddings, self.o_embeddings, self.a, self.b, self.c, self.d, M_prior=self.M_prior)
            # Safe tensor aggregation - sum all loss components
            final_loss = torch.tensor(0.0, device=v_o_bar.device)
            for key in losses:
                if torch.is_tensor(losses[key]) and losses[key].numel() > 0:
                    final_loss += self.theta[key] * losses[key]
            final_loss.backward()
            self.optimizer.step()
            # Clamp MP tensor values to [0, 1]
            with torch.no_grad():
                self.model.MS_P.data.clamp_(min=0.0, max=1.0)
                self.model.MA_P.data.clamp_(min=0.0, max=1.0)
                
                
        _, final_MP = self.model(self.vi_tensor)
        return final_MP.detach()
    

    # --- DFOL Test Function (no dataloader, uses trainer tensors) ---
    def test(self):
        self.model.eval()
        with torch.no_grad():
            v_o_bar, MP = self.model(self.vi_tensor)
            losses = self.loss_fn.forward(MP, self.vo_tensor, v_o_bar, self.b_embeddings, self.o_embeddings, self.a, self.b, self.c, self.d, M_prior=self.M_prior)
            test_loss = losses["LI"].item()  # Use inference loss for reporting
            # For binary classification, threshold at 0.5
            v_o_bar_flat = v_o_bar.squeeze(1)  # Match vo tensor shape
            pred = (v_o_bar_flat > 0.5).float()
            correct = (pred == self.vo_tensor).type(torch.float).sum().item()
            size = self.vo_tensor.shape[0]
            accuracy = correct / size

            
            
    def generate_rule_filters(self, MP, feature_names):
        """
    Applies multiple rule filters (τf) to the trained matrix MP to generate 
    a set of candidate symbolic rules (R̃).
    
    Args:
        MP (torch.Tensor): The trained SH matrix (m x C).
        feature_names (list): List of symbolic names for body features.
        
    Returns:
        list: A list of candidate rules, each containing (rule_string, rule_filter_value).
    """
    
        # τf range: from 0 to 1 with step 0.05 [1]
        rule_filters = np.arange(0.0, 1.01, 0.05)
        
        m, C = MP.shape
        candidate_rules = []
        
        # Iterate through each rule (row k) in MP
        for k in range(m):
            rule_row = MP[k, :]
            
            # Iteratively apply each τf 
            for tau_f in rule_filters:
                # Select valid features whose values are greater than τf [1]
                valid_indices = torch.where(rule_row > tau_f).tolist()
                
                if valid_indices:
                    body_atoms = [feature_names[i] for i in valid_indices]
                    
                    # Assuming a generic head atom 'Target(X, Y)' for symbolic representation
                    # This step represents forming the symbolic rule: head <- body(r) [1]
                    rule_body = " & ".join(body_atoms)
                    rule_string = f"Target(X,Y) :- {rule_body}."
                    
                    # Add the generated rule to the set R̃ 
                    candidate_rules.append({
                        'rule_string': rule_string,
                        'MP_row_index': k,
                        'tau_f': tau_f
                    })
                    
        return candidate_rules
