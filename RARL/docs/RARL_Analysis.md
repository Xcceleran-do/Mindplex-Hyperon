
# Paper Analysis: Rule-Aware Reinforcement Learning for Knowledge Graph Reasoning

**Paper**: Rule-Aware Reinforcement Learning for Knowledge Graph Reasoning  
**Authors**: Zhongni Hou, Xiaolong Jin, Zixuan Li, Long Bai  
**Affiliations**: University of Chinese Academy of Sciences; CAS Key Laboratory of Network Data Science and Technology  
**Conference**: Findings of ACL-IJCNLP 2021  
**Pages**: 4687–4692  

---

## Problem Statement

Multi-hop reasoning with Reinforcement Learning (RL) is effective and interpretable for Knowledge Graph (KG) completion. However, applying RL directly in large KGs faces two major issues:

### 1. Sparse Reward Problem
- The number of possible paths grows exponentially.
- Most paths don’t end at a correct answer → rare positive reward.
- Early exploration behaves like a random walk.
- Learning becomes slow due to lack of feedback.

### 2. Spurious Path Problem
- RL agents reinforce any path that leads to a correct entity.
- Spurious paths are semantically meaningless but reach the right answer.
- These paths distort the learning process and reduce interpretability.

**Goal**: Overcome both problems without sacrificing interpretability by using symbolic rules and strategic exploration.

---

## RARL Overview

RARL integrates symbolic rules into the RL reasoning process and uses a **three-stage action selection strategy** within a **partially random beam search** framework. This improves the probability of finding meaningful paths (reducing sparse rewards) and prevents overfitting to spurious high-scoring paths.

### Key Features:

- **Rule Injection**: Symbolic rules guide the agent toward meaningful paths, increasing reward signals. which addresses the sparse reward problem by increasing the probability of the agent finding the correct path and receiving positive reward. 
- **Partially Random Beam Search**: Prevents overfitting to high-score spurious paths. instead of selecting the highest scoring path, a fraction of the beam get's populated by randomly sampling from all possible candidates which prevents the model from becoming fixated on high scoring semantically meaningless paths. 
- **Three-Stage Action Selection Strategy**:  
  1. Random sampling  
  2. Rule-based filtering  
  3. Score-based selection  

---

### Knowledge Graph Representation

```math
\mathcal{G} = (e_s, r, e_t) \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}
```

- (E) : Entities (nodes)  
- (R): Relations (edges)  
- Query format: ( (e_s, r_q, ?) ) → predict \( e_t \)

---

### Policy Network

```math
\pi_\theta(a_t|e_t) = \sigma(\tilde{A}_t (W_2 \cdot \text{ReLU}(W_1 [h_t; e_t; r_q])))
```

- Outputs action probabilities.
- Uses LSTM-encoded history \( h_t \), current entity \( e_t \), and query \( r_q \).

### Pseudocode

```python
import torch
import torch.nn as nn


class PathEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=3):
        super(PathEncoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, path_embeddings):
        _, (hidden_state, _) = self.lstm(path_embeddings)
        return hidden_state[-1]

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_space_dim):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, action_space_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state_tensor):
        x = self.relu(self.layer1(state_tensor))
        action_probs = self.softmax(self.layer2(x))
        return action_probs

```

### Reward
- Binary: 1 if a path ends at correct entity \( e_t \), otherwise 0.

---

## Implementation

**Prerequisites**

- Python 3.8+
- A standard deep learning framework (e.g., PyTorch, TensorFlow)
- The random library (standard)
- AMIE+ (or a similar rule mining tool) for extracting symbolic rules from the knowledge graph.

---

### Initialization and Setup

- **KG Triples**: oad the KG triples (subject, relation, object). This can be stored in any efficient graph data structure that allows quick retrieval of outgoing edges from any given entity.

- **Rule Pool**: Load the pre-mined symbolic rules. These should be stored in a way that allows for efficient lookup based on a query relation (e.g., a dictionary where keys are relation heads). The rules take the form head <- body, where the body is a sequence of relations. The paper notes using AMIE+ to extract rules with a maximum length of 2. 
 
- **Embeddings**: Initialize numerical vector embeddings for all entities and relations in the KG. Dimension between 50 and 200.  

- **LSTM Path Encoder**: Set up a Long Short-Term Memory (LSTM) network which encodes the agent’s path history (a sequence of entities and relations) into a fixed-size vector at each step. 3 layers, hidden size 100–200.
- **Policy Network**: Implement a two-layered feed-forward neural network which will take the concatenated embeddings of the current path history, the current entity, and the original query relation as input \( h_t, e_t, r_q \) . It outputs a probability distribution over all possible actions (outgoing edges) from the current entity.

---

### Rule-Based Action Selection 

- The Rule-Based mechanism of the RARL is designed to combat the sparse reward problem. In a vast KG, an agent exploring randomly is highly unlikely to find a correct path and receive a reward. To make the exploration more efficient, RARL injects high-quality, human-readable symbolic rules as prior knowledge to guide the agent. 

Rules are Horn clauses:

```
head_relation(x0, xN) ← r1(x0,x1) ∧ r2(x1,x2) ... ∧ rn(xn-1,xn)
```

Example:

```
nationality_of(x,z) ← born_in(x,y) ∧ city_in(y,z)
```

- Note: r(xi , xj) is equivalent to the fact triple (xi , r, xj).

***Matching Process***

1. For query relation `rq`, retrieve all rules with head `rq`.
2. For each candidate path, extract its relation sequence.
3. Check if it matches a prefix of any rule body.
4. Keep only matching candidates.

***Random Masking***

Randomly ignore some applicable rules each step to maintain exploratory capability.

### Pseudocode

``` python 

def get_path_relations(path):
    # Path is a list of entities and relations, e.g., [e1, r1, e2, r3, e4]
    return [path[i] for i in range(1, len(path), 2)]

def rule_based_selection(candidate_paths, query_relation, rule_pool):
    """
    Filters candidate paths to find those that are rule-compliant.
    """
    rule_compliant_paths = # a list that stores all the candidate paths (from the pool of non-randomly selected candidates) that match a predefined symbolic rule.
    other_paths = # a list that stores all the candidate paths that were not selected randomly and also do not match any of the symbolic rules. These paths represent the agent's "free exploration" choices, guided solely by the scores from the policy network.
    
    if query_relation not in rule_pool:
        return, candidate_paths

    relevant_rules = rule_pool[query_relation]
    
    for candidate in candidate_paths:
        path_relations = get_path_relations(candidate['path'])
        is_compliant = False
        for rule_body in relevant_rules:
            # Check if the path's relation sequence is a prefix of the rule's body
            if len(path_relations) <= len(rule_body) and path_relations == rule_body[:len(path_relations)]:
                is_compliant = True
                break
        if is_compliant:
            rule_compliant_paths.append(candidate)
        else:
            other_paths.append(candidate)
            
    return rule_compliant_paths, other_paths

```

---



### Partially Random Beam Search

This mechanism addresses the spurious path problem by forcing diversity into the search. Instead of a purely greedy selection, a portion of the beam is reserved for random exploration. 

**Process**

1. From all candidates, randomly select `λK` paths (with replacement if necessary).
2. The rest are chosen via rule-based and score-based selection.

***Pseudocode***

``` python 
def partially_random_beam_search(all_candidates, K, lambda_factor):
    """
    Selects the first part of the beam by random sampling.
    """
    num_random_slots = round(lambda_factor * K)
    
    # Sampling with replacement if the pool is smaller.
    can_replace = len(all_candidates) < num_random_slots
    
    if can_replace:
        random_part = random.choices(all_candidates, k=num_random_slots)
    else:
        random_part = random.sample(all_candidates, num_random_slots)
        
    remaining_candidates = [p for p in all_candidates if p not in random_part]
    
    return random_part, remaining_candidates
```

---

### Overall Selection Strategy

Combines the two mechanisms into a **three-stage process**:

1. **Random Sampling** — Select `λK` candidates randomly.
2. **Rule-Based Selection** — From the remainder, choose up to `(1-λ)K` rule-compliant candidates.
3. **Score-Based Fill** — Fill any leftover slots with highest-scoring remaining candidates.

***Pseudocode***

``` python 

def overall_selection_strategy(all_candidates, K, lambda_factor, query_relation, rule_pool):
    """
    Implements the full three-stage selection strategy of RARL.
    """
    # Stage 1: Random Sampling
    random_part, remaining_candidates = partially_random_beam_search(all_candidates, K, lambda_factor)
    
    next_beam = random_part
    num_greedy_slots = K - len(next_beam)

    if num_greedy_slots <= 0:
        return next_beam

    # Stage 2: Rule-Based Selection
    rule_compliant_paths, other_paths = rule_based_selection(remaining_candidates, query_relation, rule_pool)
    
    # Stage 3: High-Score Selection
    final_greedy_selection = [] 
    # The final list of paths that will fill the "greedy" portion of the beam, which has a size of (1-λ)K
    
    if len(rule_compliant_paths) >= num_greedy_slots:
        # If there are enough (or more) rule-compliant paths, select the best from them.
        rule_compliant_paths.sort(key=lambda x: x['score'], reverse=True)
        final_greedy_selection = rule_compliant_paths[:num_greedy_slots]
    else:
        # If there are not enough rule-compliant paths, take all of them...
        final_greedy_selection.extend(rule_compliant_paths)
        num_needed_from_others = num_greedy_slots - len(rule_compliant_paths)
        
        #...and fill the rest with the highest-scoring non-rule paths.
        if num_needed_from_others > 0:
            other_paths.sort(key=lambda x: x['score'], reverse=True)
            final_greedy_selection.extend(other_paths[:num_needed_from_others])
            
    next_beam.extend(final_greedy_selection)
    
    return next_beam

```
---

### Main Training Loop

The training process is executed over many episodes. For each training query of the form (e_s, r_q,?):

1. **Initialize beam** with the start entity.
2. **Iterate for T steps**:
   - Expand all candidates from the current beam.
   - Score them using the policy network.
   - Apply **Overall Selection Strategy** to form the next beam.
3. **Reward assignment**: 1 if target entity is reached, else 0.
4.  **Optimize Policy:** The model's objective is to maximize the expected reward, formulated as:
```math
J(\theta) = \sum_{(e_s, r_q, e_t) \in G}    \sum_{p \in P(e_s, r_q)} R(p) \pi_{\theta}(p)
```
 
It uses the collected rewards to update the weights of the policy network ($\pi_{\theta}$) using the **REINFORCE** algorithm. This policy gradient method adjusts the network to increase the probability of taking actions that lead to positive rewards.


***Pseudocode**

```python

for episode in range(num_episodes):
    for query in training_queries:
        start_entity, query_relation, target_entity = query
        beam = [([start_entity], 0.0)]

        for t in range(max_path_length):
            all_candidates = generate_candidates(beam, kg)
            scored_candidates = score_paths(all_candidates, policy_net, path_encoder, query_relation)
            beam = rarl_action_selection(scored_candidates, K, LAMBDA, rules, query_relation)

        rewards = calculate_rewards(beam, target_entity)
        update_policy(policy_net, beam, rewards)
```

---


### Hyperparameters

- **K**: Beam size.
- **λ**: Randomness factor (0.4–0.9 depending on dataset).
- **T**: Max hops per episode.
- **Embedding size**: 50–200.
- **LSTM layers**: 3, hidden size 100–200.

---

## Summary 

RARL improves RL-based KG reasoning by:

- **Injecting symbolic rules** to guide exploration.
- **Adding controlled randomness** to avoid overfitting to spurious paths.
- **Using a structured three-stage selection** for balanced exploration and exploitation.

This results in higher accuracy and better interpretability compared to standard multi-hop reasoning models.


