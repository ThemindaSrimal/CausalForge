# CausalForge Phase 3 Evaluation

Evaluation date: October 31, 2026

Model: `muse-spark-1.3`

Protocol: 10 locked procedural seeds, 10 real Muse-directed experiments per seed

## Success definition

A world passes when the local model:

1. Exactly identifies both active causal factors; and
2. Achieves coefficient RMSE at or below 0.25 across all four factors.

The phase target was at least 8 successful worlds out of 10, with no incomplete runs.

## Results

| Seed | Exact factors | Coefficient RMSE | Result |
|---:|:---:|---:|:---:|
| 104729 | Yes | 0.022411 | Pass |
| 130363 | Yes | 0.019915 | Pass |
| 155921 | Yes | 0.016733 | Pass |
| 196613 | Yes | 0.013974 | Pass |
| 225307 | Yes | 0.012017 | Pass |
| 262147 | Yes | 0.024229 | Pass |
| 299993 | Yes | 0.018609 | Pass |
| 350377 | Yes | 0.025722 | Pass |
| 410117 | Yes | 0.021532 | Pass |
| 499979 | Yes | 0.015625 | Pass |

## Aggregate outcome

- Completed runs: **10/10**
- Failed runs: **0**
- Successful runs: **10/10**
- Success rate: **100%**
- Target met: **Yes**
- Measured evaluation spend: **$0.71795890**
- Mean coefficient RMSE: **0.019477**

Detailed machine-readable metrics are generated at `runs/evaluation-report.json`. Per-experiment evidence, including Muse response IDs, interventions, observations, learner updates, token usage, latency, and cost, is generated under `runs/evaluation/`.
