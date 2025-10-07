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

        v_o_bar_flat = v_o_bar.squeeze(1)  # (batch_size,)
        print(v_o_bar, v_o_bar_flat)
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
                
            # Print progress every 20 epochs to see convergence
            # if epoch % 20 == 0:
            #     v_o_bar_flat = v_o_bar.squeeze(1)
            #     print(f"Epoch {epoch}: Loss={final_loss.item():.4f}, LI={losses['LI'].item():.4f}")
            #     print(f"  Predictions: {v_o_bar_flat.detach().numpy()}")
            #     print(f"  Targets:     {self.vo_tensor.detach().numpy()}")
                
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
            # print(f"DFOL Test: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

            
            
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

# # dfol_grandparent_fixed.py
# # Ready-to-run DFOL-style implementation for the grandparent task
# # Requirements: torch, numpy

# import itertools
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# torch.manual_seed(0)
# np.random.seed(0)

# # ----------------------------
# # 1) Construct grandparent dataset
# # ----------------------------
# E = ["alice", "bob", "carol", "dave", "erin"]

# parent_facts = {
#     ("mother", "alice", "bob"),
#     ("father", "bob", "carol"),
#     ("father", "carol", "dave"),
#     ("mother", "alice", "erin"),
# }

# gp_positives = {
#     ("gp", "alice", "carol"),
#     ("gp", "bob", "dave"),
#     ("gp", "alice", "dave"),
# }

# B = set(parent_facts)
# P = set(gp_positives)

# print(f"Background parent facts: {len(parent_facts)}, gp positives: {len(gp_positives)}")

# # ----------------------------
# # 2) Propositionalization
# # ----------------------------
# binary_preds = ["father", "mother", "gp"]
# V_vars = ["X", "Y", "Z"]  # includes depth-1 variable Z

# # PF = all possible parent atoms (exclude target gp)
# PF = []
# for p in binary_preds:
#     if p == "gp":
#         continue  # exclude target predicate
#     for a, b in itertools.permutations(V_vars, 2):
#         PF.append((p, a, b))
# PF = list(dict.fromkeys(PF))
# C = len(PF)
# print("Valid body features (PF):")
# for i, f in enumerate(PF):
#     print(i, f)
# print("Total features C =", C)

# # Domains
# domain_X = sorted({ex[1] for ex in P})
# domain_Y = sorted({ex[2] for ex in P})
# domain_Z = list(E)

# # substitutions S = X × Y × Z
# substitutions = list(itertools.product(domain_X, domain_Y, domain_Z))

# def fact_true(atom):
#     return atom in B or atom in P

# # Build training dataset
# T_inputs, T_outputs = [], []
# for (x_val, y_val, z_val) in substitutions:
#     vi = np.zeros(C, dtype=np.float32)
#     for i, feat in enumerate(PF):
#         p, a, b = feat
#         a_val = {"X": x_val, "Y": y_val, "Z": z_val}[a]
#         b_val = {"X": x_val, "Y": y_val, "Z": z_val}[b]
#         if fact_true((p, a_val, b_val)):
#             vi[i] = 1.0
#     vo = np.array([1.0 if ("gp", x_val, y_val) in P else 0.0], dtype=np.float32)
#     if vi.sum() > 0:
#         T_inputs.append(vi)
#         T_outputs.append(vo)

# T_inputs = np.stack(T_inputs)
# T_outputs = np.stack(T_outputs)
# print("Train dataset size:", T_inputs.shape[0])

# # ----------------------------
# # 3) Model
# # ----------------------------
# device = torch.device("cpu")
# m = 3  # number of candidate rules
# MP_raw = nn.Parameter(torch.randn(m, C) * 0.1)
# gamma = 10.0
# optimizer = optim.Adam([MP_raw], lr=0.01)

# X_train = torch.tensor(T_inputs, dtype=torch.float32, device=device)
# Y_train = torch.tensor(T_outputs, dtype=torch.float32, device=device)

# bce_loss = nn.BCELoss()
# mse_loss = nn.MSELoss()

# # Build basic embeddings b(alpha) for X,Y head vars
# basic_embeddings = []
# for feat in PF:
#     _, a, b = feat
#     bvec = [1 if a != "Z" else 0, 1 if b != "Z" else 0]
#     basic_embeddings.append(bvec)
# basic_embeddings = torch.tensor(basic_embeddings, dtype=torch.float32, device=device)

# def fuzzy_or(xs): return 1 - torch.prod(1 - xs, dim=-1)
# def phi(z): return torch.sigmoid(gamma * z)

# def infer_vbar(MP_sigmoid, Xi):
#     dots = Xi @ MP_sigmoid.t()
#     z = dots - 1.0
#     rule_vals = phi(z)
#     vbar = 1 - torch.prod(1 - rule_vals, dim=1, keepdim=True)
#     return vbar

# def basic_loss(MP_sigmoid):
#     Mk_b = MP_sigmoid[:, :, None] * basic_embeddings[None, :, :]
#     or_per_kj = 1.0 - torch.prod(1.0 - Mk_b, dim=1)
#     basick = torch.prod(or_per_kj, dim=1)
#     return mse_loss(basick, torch.ones_like(basick))

# def rowsum_loss(MP_sigmoid):
#     return mse_loss(torch.sum(MP_sigmoid, dim=1), torch.ones(m, device=MP_sigmoid.device))

# # Occurrence loss: encourage all variables {X,Y,Z} to appear
# var_sets = []
# for feat in PF:
#     _, a, b = feat
#     var_sets.append({a, b})

# def occurrence_loss(MP_sigmoid):
#     losses = []
#     for k in range(MP_sigmoid.shape[0]):
#         cover = {v: 0.0 for v in V_vars}
#         for i, vars_in_feat in enumerate(var_sets):
#             for v in vars_in_feat:
#                 cover[v] = max(cover[v], MP_sigmoid[k, i].item())
#         losses.append(np.prod(list(cover.values())))
#     return 1 - torch.tensor(losses, dtype=torch.float32, device=MP_sigmoid.device).mean()

# # ----------------------------
# # 4) Training
# # ----------------------------
# num_epochs = 5000
# for epoch in range(1, num_epochs+1):
#     optimizer.zero_grad()
#     MP_sigmoid = torch.sigmoid(MP_raw)
#     vbar = infer_vbar(MP_sigmoid, X_train)
#     LI = bce_loss(vbar, Y_train)
#     LS = rowsum_loss(MP_sigmoid)
#     LB = basic_loss(MP_sigmoid)
#     LO = occurrence_loss(MP_sigmoid)
#     loss = LI + 0.5 * LS + 10 * LB + 10 * LO
#     loss.backward()
#     optimizer.step()
#     if epoch % 500 == 0:
#         acc = ((vbar.detach().cpu().numpy() > 0.5).astype(int).squeeze()
#                == Y_train.cpu().numpy().squeeze()).mean()
#         print(f"Epoch {epoch} loss={loss.item():.4f} acc={acc:.3f}")

# # ----------------------------
# # 5) Extraction
# # ----------------------------
# MP_final = torch.sigmoid(MP_raw).detach().cpu().numpy()
# taus = np.arange(0.0, 1.01, 0.1)

# def apply_candidate_rule(body_feats):
#     supports, correct = 0, 0
#     for (x_val, y_val, z_val) in substitutions:
#         if all((p, {"X":x_val,"Y":y_val,"Z":z_val}[a],
#                      {"X":x_val,"Y":y_val,"Z":z_val}[b]) in B
#                for (p,a,b) in body_feats):
#             supports += 1
#             if ("gp", x_val, y_val) in P:
#                 correct += 1
#     return (correct / supports if supports else 0.0), supports

# candidates = []
# for k in range(m):
#     for tau in taus:
#         body = [PF[i] for i,v in enumerate(MP_final[k]) if v > tau]
#         if not body: continue
#         prec, sup = apply_candidate_rule(body)
#         candidates.append((prec, sup, body))

# sound = [ (prec, sup, tuple(body)) for (prec, sup, body) in candidates if prec >= 0.999 and sup > 0 ]
# sound = list(set(sound))
# print("\nExtracted sound rules:")
# for prec, sup, body in sound:
#     print(f"gp(X,Y) :- {body}   (prec={prec:.3f}, support={sup})")

