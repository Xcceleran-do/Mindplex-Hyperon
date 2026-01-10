# Performance Benchmarks: Pattern Mining

## Test Environment

**System Specifications**
- Platform: GitHub Codespace
- CPU: 4-core
- RAM: 16 GB
- Storage: 32 GB

**Dataset**
- Source: `experiments/atomspace_visualizer/public/data.metta` (demo-main branch)
- Size: 180 facts
- Attributes per fact: length, reading-time, date-period, category, popularity, engagement, authored-by, title

---

## Benchmark Results

### Hyperon-Experimental (HE)
**Branch**: demo-main  
**MeTTa Version**: 0.2.9

| Conjunct | Patterns Found | Real Time | User Time | Sys Time | Status |
|----------|---|-----------|-----------|----------|--------|
| 3 | 9 | 1m 18.533s | 1m 17.786s | 0m 0.717s | ✅ Complete* |
| > 3 | — | — | — | — | ❌ Crashes |

**Notes**
- Requires filter function to exclude "audience-expertise" and "engagement" predicates from conjunct generation
- Crashes without filter even at conjunct=3
- Frequency (min support) = 3

---

### PeTTa (Prolog-Based MeTTa)
**Branch**: PeTTa-ASI  
**MeTTa Runtime**: PeTTa (SWI-Prolog backend)

| Conjunct | Patterns Found | Real Time | User Time | Sys Time | Frequency |
|----------|---|-----------|-----------|----------|-----------|
| 3 | 114 | 0m 1.579s | 0m 1.446s | 0m 0.034s | 3 |
| 4 | 80 | 0m 26.318s | 0m 26.168s | 0m 0.050s | 3 |
| 5 | 35 | 4m 44.846s | 4m 44.581s | 0m 0.151s | 5 |

**Notes**
- No filtering required; all predicates handled
- Scales to conjunct=5 without crashes
- Exponential time growth with conjunct size

---

## Key Findings

### Performance Comparison (Conjunct=3)

| Metric | HE | PeTTa | Speedup |
|--------|----|----|---------|
| Real Time | 78.5s | 1.6s | **~49x faster** |
| Patterns Found | 9 | 114 | ~13x more patterns |

### Scalability

- **HE**: Limited to conjunct≤3 (with filtering); unstable without predicates filter
- **PeTTa**: Scales to conjunct≥5; exponential slowdown but stable
  - conjunct 3→4: ~16.7x slower
  - conjunct 4→5: ~10.8x slower

### Memory & Stability

- **HE (RAM-based AtomSpace)**: Out-of-memory crashes above conjunct=3; filter workaround required
- **PeTTa (Prolog/Indexing)**: Memory-efficient; completes larger searches without crashes

---

## Recommendations

### When to Use Each Implementation

**Use PeTTa for:**
- Larger datasets (>500 facts)
- Higher conjunction depths (conjunct ≥ 4)
- Production pattern mining requiring stability
- Exploratory analysis beyond test datasets

**Use HE for:**
- Small datasets with limited attributes
- Integration with Python/AI chat workflows
- Visualization backends (currently integrated with visualizer)
- Real-time interactive queries (if conjunct ≤ 3)

### Optimization Strategies

1. **Data Filtering**: HE requires predicate filtering to avoid state explosion
2. **Staged Mining**: Mine with conjunct=3 first, then higher depths separately
3. **Minimum Support**: Increase `minsup` parameter to reduce pattern space
4. **Sampling**: For datasets >500 facts, consider stratified sampling before mining

---

## Future Work

- Hybrid approach: Use PeTTa for initial pattern discovery → Feed to HE for visualization
- Memory-mapped AtomSpace for HE to support larger datasets
- Parallel pattern mining across conjunct depths
- Caching of intermediate pattern results

---

## How to Reproduce

### HE Benchmark
```bash
cd /workspaces/Mindplex-Hyperon/PeTTa
time sh run.sh ../experiments/pattern-miner/tests/test-pattern-miner.metta
```

### PeTTa Benchmark
```bash
cd /workspaces/Mindplex-Hyperon/PeTTa
time sh run.sh ../path/to/test.metta  # PeTTa test file with conjunct iterations
```

### Dataset Location
```
experiments/atomspace_visualizer/public/data.metta
```

---

## Test Date
January 10, 2026

