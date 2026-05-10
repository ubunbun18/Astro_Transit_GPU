"""
n_db deduplication analysis:
Measure how many unique n_db values exist per period across the full search grid.
"""
import numpy as np

N_BINS = 128
durations = np.linspace(0.01, 0.2, 8)
periods = np.linspace(0.5, 13.0, 100000)

total_evals = 0
deduped_evals = 0

for p in periods:
    n_db_arr = np.round(durations * N_BINS / p + 0.5).astype(int)
    n_db_arr = np.maximum(n_db_arr, 1)
    total_evals += len(durations)
    deduped_evals += len(np.unique(n_db_arr))

reduction = (1 - deduped_evals / total_evals) * 100
print(f"Total evaluations (original):     {total_evals:,}")
print(f"Total evaluations (deduplicated): {deduped_evals:,}")
print(f"Redundant computation:            {reduction:.1f}%")
print(f"Expected speedup on inner loop:   {total_evals/deduped_evals:.2f}x")

# Per period range
for p_lo, p_hi in [(0.5,2), (2,5), (5,10), (10,13)]:
    mask = (periods >= p_lo) & (periods < p_hi)
    p_sub = periods[mask]
    avg_unique = np.mean([len(np.unique(np.maximum(np.round(durations*N_BINS/p+0.5).astype(int),1))) 
                          for p in p_sub])
    print(f"  P=[{p_lo:.0f},{p_hi:.0f}d): avg unique n_db = {avg_unique:.2f} / 8")
