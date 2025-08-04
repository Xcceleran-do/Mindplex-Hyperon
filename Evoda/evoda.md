# Analysis of "Rule Learning Over Knowledge Graphs With Genetic Logic Programming"

[Paper Link](https://www.scribd.com/document/774579430/Rule-Learning-over-Knowledge-Graphs-with-Genetic-Logic-Programming)

## Overview

This paper presents an evolutionary algorithm for logic rule mining over knowledge graphs (KGs), leveraging common genetic operators: **selection**, **mutation**, and **crossover**. The algorithm uses **Vadalog** for knowledge representation and defines the rule learning problem as: _Given a KG and a target predicate, find a rule with the target in its head (using Horn rule notation)_.

## Rule Evaluation

### 1. Internal Evaluation (Fitness Functions)

- **Standard Confidence**:  
    Measures the reliability of a rule under the Closed World Assumption (CWA), where unknown facts are considered false.
    - **Body Support**: Number of instances in the KG matching the rule's body.
    - **Rule Support**: Number of instances where both the body and head are satisfied.
    - **Formula**:  
        `standard_confidence = rule_support / body_support`

- **PCA Confidence**:  
    Addresses KG incompleteness using the Partial Completeness Assumption (PCA). Only considers entities for which at least one head fact is known.
    - **Formula**:  
        `pca_confidence = rule_support / pca_body_support`

#### Pseudocode: Fitness Calculation
```pseudo
function standard_confidence(rule, KG):
        body_support = count_matches(rule.body, KG)
        rule_support = count_matches(rule.body + rule.head, KG)
        return rule_support / body_support

function pca_confidence(rule, KG):
        pca_body_support = count_pca_matches(rule, KG)
        rule_support = count_matches(rule.body + rule.head, KG)
        return rule_support / pca_body_support
```

### 2. External Evaluation

- Performed manually, as it operates on facts not present in the given KG.

## Core Algorithm

### A. Rule Discovery

#### 1. Rule Transformation Operators

- **Selection**: Selects one or two individuals from the population based on fitness-proportional selection.
- **Mutation**: Alters predicates or variables in a rule with a given mutation rate.
- **Crossover**: Swaps atoms between two rules' bodies with a given crossover rate.

#### Pseudocode: Genetic Operators
```pseudo
function select(population, fitness_fn):
        fitnesses = [fitness_fn(ind) for ind in population]
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]
        return sample(population, probabilities, k=2) #or k=1

function mutate(rule, mutation_rate, predicates, variables):
        for atom in rule.atoms:
                if random() < mutation_rate:
                        atom.predicate = random_choice(predicates)
                for var in atom.variables:
                        if random() < mutation_rate:
                                var = random_choice(variables)
        return rule

function crossover(rule1, rule2, crossover_rate):
        if random() < crossover_rate:
                atom1 = random_choice(rule1.body)
                atom2 = random_choice(rule2.body)
                rule1 = rule1[0: before atom1] + atom2 + rule1[after atom1:]
                rule2 = rule2[0: before atom2] + atom1 + rule2[after atom2:]
        return rule1, rule2
```

#### 2. Rule Learning Algorithm

- Initializes a population (possibly using singleton copy rules: rules with only the target predicate in the head and a single atom in the body which is a copy of the target).
- Iteratively applies selection, crossover, and mutation to evolve rules.
- Maintains a set of unique, high-fitness rules.

##### Pseudocode: Rule Learning
```pseudo
function rule_learning(KG, target, population_size, max_generations, fitness_fn, mutation_rate, crossover_rate):
        population = initialize_population(target, population_size)
        best_rules = argmax(populaton, fitness_fn) 
        for generation in range(max_generations):
                while len(best_rules) < population_size:
                        parent1, parent2 = select(population, fitness_fn)
                        child1, child2 = crossover(parent1, parent2, crossover_rate)
                        child1 = mutate(child1, mutation_rate)
                        child2 = mutate(child2, mutation_rate)
                        if child1 not in best_rules:
                                best_rules.append(child1)

                        if child2 not in best_rules:
                                best_rules.append(child1)
```
#### 3. Rule Selection

- The new rule must:
        - Not be a duplicate of any existing rule.
        - Produce at least one new fact (i.e., infer something not already in the KG).
        - Have at least one support (i.e., be applicable to at least one instance in the KG).

---

### B. Rule Covering Algorithm

After generating a rule, it may not cover all facts in the KG. To ensure broader coverage, the algorithm iteratively learns rules and prunes covered facts until the KG is empty or a desired number of rules is reached.

##### Pseudocode: Rule Covering
```pseudo
function rule_covering(KG, target, max_rules, ...):
        rules = []
        uncovered_facts = KG.copy()
        while len(uncovered_facts) > 0 and len(rules) < max_rules:
                rule = rule_learning(uncovered_facts, target, ...)
                predicted_facts = infer_facts(rule, uncovered_facts)
                # Prune predicted facts from uncovered_facts
                uncovered_facts = uncovered_facts - predicted_facts
                rules.append(rule)
        return rules
```

## Implementation Notes

The described algorithm can be implemented and used directly without requiring adaptations at this stage. For future improvements, concepts from hypergraph theory could be incorporated to enhance rule learning and representation.

## Summary

- The algorithm evolves rules using genetic operators.
- Rule selection ensures novelty and utility.
- The covering process iteratively learns rules to maximize KG coverage.
- This approach balances rule quality and KG completeness.
