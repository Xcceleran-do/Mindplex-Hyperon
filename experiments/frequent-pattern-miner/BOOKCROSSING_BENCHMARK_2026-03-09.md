# Book Recommendation Dataset Benchmark (2026-03-09)

This document records observed runtime and rule-count outputs for frequent pattern mining on the Kaggle Book Recommendation dataset (Book-Crossing).

## Scope

- Module: `experiments/frequent-pattern-miner`
- Date: `2026-03-09`
- Dataset scale points tested: `30k`, `50k`, `70k`, `169k` data points
- Main variable changed per run: `minimum support`

## Raw Benchmark Results

The following values are copied from the benchmark session logs.

| Data points | Min support | Rules found | Real time | User time | Sys time |
|---|---:|---:|---:|---:|---:|
| 30,000 | 40 | 1,074 | 5m0.713s | 4m25.421s | 0m23.003s |
| 30,000 | 400 | 406 | 3m36.886s | 3m35.593s | 0m1.001s |
| 50,000 | 800 | 259 | 10m22.948s | 10m21.247s | 0m0.990s |
| 70,000 | 800 | 956 | 23m18.735s | 23m14.349s | 0m2.699s |
| 70,000 | 900 | >948 | 26m57.666s | 26m53.370s | 0m2.845s |
| 169,000 | 8,000 | 0 | 136m15.791s | 130m19.402s | 0m4.512s |
| 169,000 | 900 | >924 | 163m38.960s | 118m57.383s | 0m8.689s |

## Normalized Runtime (Real Seconds)

| Data points | Min support | Rules found | Real seconds |
|---|---:|---:|---:|
| 30,000 | 40 | 1,074 | 300.713 |
| 30,000 | 400 | 406 | 216.886 |
| 50,000 | 800 | 259 | 622.948 |
| 70,000 | 800 | 956 | 1,398.735 |
| 70,000 | 900 | >948 | 1,617.666 |
| 169,000 | 8,000 | 0 | 8,175.791 |
| 169,000 | 900 | >924 | 9,818.960 |

## Quick Observations

1. Runtime generally increases strongly with data size.
2. At `30k`, increasing support from `40` to `400` reduced both runtime and output size.
3. At `169k`, very high support (`8000`) produced no rules, but runtime remained high, indicating substantial candidate/search work still occurs before pruning.
4. The `169k` runs are significantly slower than smaller-scale runs (multi-hour runtime), which is an important practical limit for interactive experimentation.
5. Rule counts are not strictly monotonic across different dataset sizes because both dataset size and support threshold changed between runs.

## Notes On Interpretation

- Some results use lower-bound notation (`>948`, `>924`). Treat these as minimum confirmed counts, not exact counts.
- Since multiple variables changed between runs (data points and support), this benchmark is best treated as exploratory rather than a controlled scaling study.
- For stronger conclusions, run a grid where only one parameter changes at a time.

## Recommended Follow-Up Benchmark Matrix

Use a fixed benchmark matrix to improve comparability:

- Data points: `30k`, `50k`, `70k`, `100k`, `169k`
- Min support: `40`, `200`, `400`, `800`, `900`, `2000`, `8000`
- Keep all other parameters fixed (depth, preprocessing, hardware, and runtime command)

For each run, record:

- command line and git commit hash
- data sampling method
- exact rules count
- real/user/sys times
- peak memory if available

## Reproducibility Template

Fill this section for future updates.

```text
Date:
Commit:
Machine (CPU/RAM/OS):
Command:
Dataset source and sampling:
Depth:
Min support:
Output rules:
real/user/sys:
Notes:
```
