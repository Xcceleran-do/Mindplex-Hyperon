# Paper Analysis: RuDiK - Rule Discovery in Knowledge Bases


**Paper**: RuDiK: Rule Discovery in Knowledge Bases
**Authors**: Stefano Ortona, Venkata Vamsikrishna Meduri, Paolo Papotti
**Affiliations**: Meltwater; Arizona State University; EURECOM

---

## Problem Statement

Knowledge Bases (KBs) frequently encounter **incompleteness** (facts are missing due to the Open World Assumption) and **errors** (noisy data propagated from automated extraction processes). RuDiK aims to automate the discovery of declarative rules to address these challenges by enriching KBs with new facts (positive rules) and identifying inconsistencies (negative rules).

---

## RuDiK Overview

RuDiK is a system designed to automatically discover both **positive rules** (which infer new facts) and **negative rules** (which identify data contradictions) to enhance KB data quality. The system is robust to existing noise and incompleteness in KBs and employs a more expressive rule language than previous approaches.

### Key Features:

* **Rule Discovery**: Automatically finds rules to add facts and detect errors.
* **Robustness**: Tolerates significant levels of data noise and incompleteness.
* **Accuracy**: Achieves high precision in both fact inference and error detection.
* **Scalability**: Utilizes incremental, disk-based algorithms suitable for large KBs.

---

## System Workflow

### Input:

* A **Knowledge Base** (KB) as a collection of `(subject, predicate, object)` triples.
* A **target predicate** chosen by the user (e.g., `couple`, `presidentOf`).

### Steps:

1.  **Example Generation**:
    * **Positive (G)**: This set comprises existing true facts for the target predicate, extracted directly from the KB.
    * **Negative (V)**: This set contains counterexamples, which are facts known to be false for the target predicate. These are generated strategically using the **Local-Closed World Assumption (LCWA)** to create **meaningful false examples**. LCWA identifies entity pairs that are semantically related (connected by other predicates) but for which the target predicate is explicitly or implicitly known to be false. For any `(x, y)` pair in `G` or `V`, RuDiK ensures all occurrences of `x` (as a subject) and `y` (as an `object`) have consistent types, preventing mixed-type examples.

2.  **Rule Mining**:
    * **Path Discovery & Rule Creation**: The system traverses the KB graph to identify relevant paths between entities. Each path can be generalized into a potential Horn rule body.
    * **Weight Calculation**: A quality `Weight` is assigned to each candidate rule. A lower weight indicates a better rule. The formula balances the rule's coverage over `G` (positive examples) and `V` (negative examples) using the $\alpha$ parameter:
        <br>
        $$\text{Weight} = \alpha \left(1 - \frac{|\text{r} \cap \text{G}|}{|\text{G}|}\right) + (1-\alpha) \left(\frac{|\text{r} \cap \text{V}|}{|\text{V}|}\right)$$
        <br>

        * **High $\alpha$**: Prioritizes **precision** (rules that cover few negative examples).
        * **Low $\alpha$**: Prioritizes **recall** (rules that cover many positive examples).

3.  **Greedy Rule Selection**: An iterative algorithm selects the most beneficial rules based on their marginal weight.

4.  **Rule Execution**:
    * **Positive Rules**: Used to infer and add new facts to the KB.
    * **Negative Rules**: Used to flag inconsistencies and identify erroneous information within the KB.

---

## 3. Algorithm 

We represent the Knowledge Base `KB` as a collection of `(subject, predicate, object)` tuples.

### Data Structures

```python

# Represents a discovered Horn Rule
class Rule:
    def __init__(self, head_predicate, body_atoms, G_coverage=None, V_coverage=None):
        """
        Represents a logical rule of the form:
        body_atoms ⇒ head_predicate(a, b)

        Args:
            head_predicate (str): Target relation (e.g., 'child')
            body_atoms (list or tuple): Each atom is a tuple (predicate, var1, var2)
            G_coverage (set): Covered (s, t) pairs from generation set
            V_coverage (set): Covered (s, t) pairs from validation set
        """
        self.head_predicate = head_predicate
        self.body_atoms = tuple(body_atoms)
        self.G_coverage = G_coverage or set()
        self.V_coverage = V_coverage or set()

    def __repr__(self):
        """
        Human-readable string representation of the rule.
        Example: hasChild(a,c) ∧ hasChild(b,c) ⇒ child(a,b)
        """
        body_str = " ∧ ".join(f"{p}({s},{o})" for (p, s, o) in self.body_atoms)
        return f"{body_str} ⇒ {self.head_predicate}(a,b)"

```

### 1. Example Generation: Populating `G` and `V`

#### 1.1. Positive Examples (G) Generation

```python
def generate_positive_examples(KB, target_predicate):
    """Extracts existing facts for target predicate
    Args:
        KB: List of (subject, predicate, object) triples
        target_predicate: Relation to analyze (e.g., 'childOf')
    Returns:
        Set of valid (subject, object) pairs
    """
    return {(s,o) for (s,p,o) in KB if p == target_predicate}
```

#### 1.2. Negative Examples (V) Generation with LCWA

This method identifies semantically related pairs `(x, y)` that are *not* positive examples, using LCWA to ensure they are likely true negatives. It focuses on entities connected by predicates other than the target, and ensures type consistency.

```python
def generate_negative_examples(KB, target_predicate):
    """Generates high-quality counter-examples using:
    1. Local-Closed World Assumption
    2. Type consistency checks
    3. Semantic relationship requirements
    """
    V = set()
    subjects = {s for (s,p,o) in KB if p == target_predicate}
    
    for x in subjects:
        # Find entities connected via other predicates
        connected_ys = {o for (s,p,o) in KB 
                       if s == x and p != target_predicate
                       and consistent_types(x, o, KB)}
        
        known_ys = {o for (s,p,o) in KB 
                   if s == x and p == target_predicate}
        
        # Add if: (1) Not positive, (2) LCWA holds, (3) Same type
        V.update((x,y) for y in connected_ys 
                if y not in known_ys
                and (has_other_relations(x, KB, target_predicate) or 
                     has_other_relations(y, KB, target_predicate)))
    return V

def consistent_types(x, y, KB):
    """Ensures x and y have uniform types"""
    x_types = {t for (s,p,t) in KB if s == x and p == 'type'}
    y_types = {t for (s,p,t) in KB if s == y and p == 'type'}
    return len(x_types) == 1 and len(y_types) == 1

def has_other_relations(entity, KB, predicate):
    """LCWA check: Entity has other relationships"""
    return len({o for (s,p,o) in KB 
              if s == entity and p == predicate}) > 0
```

### 2. Rule Mining: Path Discovery & Candidate Rule Creation

For each `(x, y)` pair in **G**, the system initiates a graph traversal from `x` to find paths that reach `y` within `max_path_length`. These paths form the basis for candidate rule bodies. This pseudocode does not consider literals for path expansion or literal comparison atoms.

```python
def discover_paths(KB, G, max_len=3):
    """Complete path discovery with type-aware pruning"""
    rule_candidates = {}
    
    for (x, y) in G:
        # Initialize with entity type constraints
        x_type = get_entity_type(KB, x)
        y_type = get_entity_type(KB, y)
        
        queue = deque([(x, [], set(), set())])  # (node, path, vars, visited)
        
        while queue:
            curr, path, vars, visited = queue.popleft()
            
            # Rule creation condition
            if curr == y and len(path) > 0:
                rule = create_rule(path, x, y)
                rule_candidates[rule] = rule_candidates.get(rule, 0) + 1
                continue
                
            if len(path) >= max_len:
                continue
                
            # Expand with type checking
            for (p, o) in get_outgoing(KB, curr):
                if (curr, p, o) in visited:
                    continue
                    
                o_type = get_entity_type(KB, o)
                if not is_valid_extension(x_type, y_type, o_type, p, path):
                    continue
                    
                new_vars = update_vars(vars, curr, o)
                if meets_join_conditions(new_vars):
                    new_visited = visited.copy()
                    new_visited.add((curr, p, o))
                    queue.append((o, path + [(p, o)], new_vars, new_visited))
                    
    return rule_candidates
```

### 3. Greedy Rule Selection (Main Discovery Algorithm)

This algorithm iteratively selects the most beneficial rules to cover positive examples (`G`), while minimizing coverage of negative examples (`V`), using an A*-like traversal implicitly by prioritizing expansions based on estimated marginal weight.

```python
def greedy_rule_selection(R, G, V, alpha):
    # R: candidate rules
    # G: generation set (positive examples)
    # V: validation set (negative examples)
    # alpha: weight parameter balancing precision vs recall

    R_opt = set()              # Final selected rules
    uncovered_G = set(G)       # Examples in G not yet covered

    while uncovered_G:
        best_rule = None
        best_gain = -float('inf')  # Initialize best marginal gain

        for r in R:
            if r in R_opt: continue  # Skip already selected rules

            # Compute marginal gain of adding rule r to current solution
            marginal = calculate_marginal_weight(R_opt, r, G, V, alpha)

            # Select rule with lowest (most negative) marginal weight
            if marginal < best_gain:
                best_gain = marginal
                best_rule = r

        # Stop if no rule improves the total weight
        if best_gain >= 0: break

        # Add best rule to solution and update uncovered examples
        R_opt.add(best_rule)
        uncovered_G -= coverage(best_rule, G)

    return R_opt


def calculate_marginal_weight(R, r, G, V, alpha):
    # Computes Δweight when adding rule r to rule set R
    before = calculate_weight(R, G, V, alpha)
    after = calculate_weight(R | {r}, G, V, alpha)
    return after - before

def is_valid_rule(rule, maxPathLen=3):
    # Ensures rule meets structural constraints:
    # - Target variables must appear ≥1 times
    # - Other variables must appear ≥2 times
    # - Rule body must not exceed maxPathLen

    vars = extract_variables(rule)
    target_vars = get_target_variables(rule.head)

    for v in vars:
        if v in target_vars:
            if vars.count(v) < 1: return False
        else:
            if vars.count(v) < 2: return False

    if len(rule.body) > maxPathLen: return False
    return True
    
def calculate_weight(rule, G, V, alpha=0.3):
    # Computes rule quality score based on:
    # - Coverage over G (recall)
    # - Coverage over V normalized by unbounded coverage (precision)

    C_G = coverage(rule, G)
    C_V = coverage(rule, V)
    U_V = unbounded_coverage(rule, V)

    term_G = 1 - (len(C_G) / len(G)) if G else 0
    term_V = len(C_V) / len(U_V) if U_V else 0

    return alpha * term_G + (1 - alpha) * term_V


def coverage(rule, examples):
    # Returns examples where rule body is fully instantiated
    return {e for e in examples if covers(rule, e)}

def unbounded_coverage(rule, examples):
    # Returns examples where required predicates exist
    return {e for e in examples if has_required_predicates(rule, e)}


def covers(rule, example):
    # Checks if rule body can be instantiated for (x, y)
    x, y = example
    query = f"""
    SELECT * WHERE {{
        {instantiate_body(rule.body, x, y)}
    }} LIMIT 1
    """
    return bool(execute_sparql(query))

def has_required_predicates(rule, example):
    # Checks if example has all predicates needed for rule body
    x, y = example
    required_preds = extract_predicates(rule.body)
    return all(has_triple(x, p) for p in required_preds)

```

### Example of rule discovery and Greedy Selection:

* **Load KB and Define Target Predicate**
* **Target Predicate:** `couple`.
* **Positive (G):** `generate_positive_examples(KB, target_predicate)`.
* **Negative (V):** `generate_negative_examples(KB, target_predicate)`.
* **Discovered Rule (via pathfinding):**
    ```prolog
    rule_candidates = discover_paths(KB, G, max_len=3)
    ```
* **Evaluate Coverage and Filter Valid Rules**
    ```python
    valid_rules = []

    for raw_rule in rule_candidates:
        rule_obj = Rule(
            head_predicate=target_predicate,
            body_atoms=raw_rule.body_atoms
        )

        # Check validity
        if not is_valid_rule(rule_obj): continue

        # Compute coverage sets
        rule_obj.G_coverage = coverage(rule_obj, G)
        rule_obj.V_coverage = coverage(rule_obj, V)

        valid_rules.append(rule_obj)
    ```
* **Coverage Evaluation for this rule:**
    * R_opt = greedy_rule_selection(valid_rules, G, V, alpha=0.3)

---

## 4. Conclusion & Potential Integration with Mindplex Hyperon

RuDiK offers an automated and robust solution for Knowledge Base curation by **discovering high-coverage positive and negative rules**. It effectively **balances precision and recall** using a weighted set cover approach and **scales to large KBs**.

The discovered rules provide high-quality training examples for **Machine Learning systems**. Specifically, RuDiK's ability to refine KB data could significantly enhance **Mindplex Hyperon**, an explainable recommendation engine built on **AtomSpace**. This integration would improve AtomSpace's data quality, directly inform Mindplex Hyperon's transparent reasoning, and guide its agent-based modeling for more trustworthy recommendations.
