
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

## Key Contributions

- **RARL Model**: Combines RL with symbolic rule guidance.
- **Rule Injection**: Symbolic rules guide the agent toward meaningful paths, increasing reward signals.
- **Partially Random Beam Search**: Prevents overfitting to high-score spurious paths.
- **Three-Stage Action Selection Strategy**:  
  1. Random sampling  
  2. Rule-based filtering  
  3. Score-based selection  

---

## System Overview

### Components
- **Agent**: Policy network with LSTM-based path encoder.
- **Environment**: The Knowledge Graph (KG).

### Action Selection Strategy
- Uses rule-based, score-based, and random action selection.

### Beam Search
- Maintains top-K scoring paths at each time step.

---

## Knowledge Graph Representation

```math
\mathcal{G} = (e_s, r, e_t) \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}
```

- (E) : Entities (nodes)  
- (R): Relations (edges)  
- Query format: ( (e_s, r_q, ?) ) → predict \( e_t \)

---

## Policy Network

```math
\pi_\theta(a_t|e_t) = \sigma(\tilde{A}_t (W_2 \cdot \text{ReLU}(W_1 [h_t; e_t; r_q])))
```

- Outputs action probabilities.
- Uses LSTM-encoded history \( h_t \), current entity \( e_t \), and query \( r_q \).

### Reward
- Binary: 1 if a path ends at correct entity \( e_t \), otherwise 0.

---

## Implementation

### Initialization and Setup

- **KG Triples**: oad the KG triples (subject, relation, object). This can be stored in any efficient graph data structure that allows quick retrieval of outgoing edges from any given entity.

- **Rule Pool**: Load the pre-mined symbolic rules. These should be stored in a way that allows for efficient lookup based on a query relation (e.g., a dictionary where keys are relation heads). The rules take the form head <- body, where the body is a sequence of relations. The paper notes using AMIE+ to extract rules with a maximum length of 2. 
 
- **Embeddings**: Initialize numerical vector embeddings for all entities and relations in the KG. Dimension between 50 and 200.  

- **LSTM Path Encoder**: Set up a Long Short-Term Memory (LSTM) network which encodes the agent’s path history (a sequence of entities and relations) into a fixed-size vector at each step. 3 layers, hidden size 100–200.
- **Policy Network**: Implement a two-layered feed-forward neural network which will take the concatenated embeddings of the current path history, the current entity, and the original query relation as input \( h_t, e_t, r_q \) . It outputs a probability distribution over all possible actions (outgoing edges) from the current entity.

### Hyperparameters
- **Beam Size (K)**: The number of top-scoring paths to keep at each step.
- **Lambda (λ)**: The fraction of the beam to be selected via random sampling. Suggested values are 0.9 for UMLS, 0.4 for WN18RR, and 0.7 for FB15K-237.
- **Max Path Length (T)**: The maximum number of steps (hops) an agent can take in one episode

### Sample Code

```python
import torch
import torch.nn as nn

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

class PathEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=3):
        super(PathEncoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, path_embeddings):
        _, (hidden_state, _) = self.lstm(path_embeddings)
        return hidden_state[-1]
```

---

## Main Training Loop

The training process is executed over many episodes. For each training query of the form $(e_s, r_q,?)$:

1.  **Initialize the beam:** Start the episode with an initial beam, $B_0$, containing a single path that consists of only the starting entity $e_s$.[1]
2.  **Iterate through time steps:** For each time step `t` from 0 to `T-1`, perform the action selection process to generate the next beam, $B_{t+1}$.[1]
3.  **Calculate Rewards:** After the final step `T`, evaluate each of the `K` paths in the final beam, $B_T$. A binary reward is assigned: $R(p) = 1$ if a path `p` ends at the correct target entity, and $R(p) = 0$ otherwise.
4.  **Optimize Policy:** The model's objective is to maximize the expected reward, formulated as:
```math
J(\theta) = \sum_{(e_s, r_q, e_t) \in G}    \sum_{p \in P(e_s, r_q)} R(p) \pi_{\theta}(p)
```
 
It uses the collected rewards to update the weights of the policy network ($\pi_{\theta}$) using the **REINFORCE** algorithm. This policy gradient method adjusts the network to increase the probability of taking actions that lead to positive rewards.


```python
# Conceptual Training Loop
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

## RARL Action Strategy

### 1. Candidate Generation and Scoring
- For every path currently in the beam B_t, find all possible one-hop extensions by identifying all outgoing edges from the path’s last entity. 
- Use policy network (πθ) to score each candidate. This score represents the probability of taking the last action given the path history.

### 2. Beam Construction

#### Stage 1: Random Sampling
- Randomly sample \( λK \) paths.
- Encourages exploration of diverse paths.

#### Stage 2: Rule-Based Selection
- Match candidate relation sequences with known rules.
- Optionally mask some rules to reduce overfitting.

#### Stage 3: High-Score Selection
- Fill remaining slots with top-scoring paths from remaining candidates.

The resulting \( B_{t+1} \) has K paths for the next step.


