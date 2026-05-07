import sys
import os
import numpy as np
import cupy as cp
import time
from astrotransit_gpu.search.cpu_reference_bls import run_astropy_bls
from astrotransit_gpu.search.gpu_bls import run_gpu_bls
from astrotransit_gpu.inject.box import inject_box_transit

import pytest

@pytest.mark.gpu
def test_accuracy():
    # 1. Create synthetic data
    t = np.linspace(0, 20, 5000)
    f = np.ones_like(t)
    
    # Inject a planet
    true_p = 5.321
    true_t0 = 1.2
    true_dur = 0.12
    true_depth = 0.005 # 0.5%
    
    f_injected = inject_box_transit(t, f, true_p, true_t0, true_dur, true_depth)
    # Add minor noise
    np.random.seed(42)
    f_injected += np.random.normal(0, 0.0005, size=len(t))
    
    # 2. Define search grid
    periods = np.linspace(2.0, 10.0, 500)
    durations = np.array([0.05, 0.1, 0.15, 0.2])
    
    # 3. Run CPU BLS
    print("Running CPU BLS (Astropy)...")
    start_time = time.time()
    cpu_res = run_astropy_bls(t, f_injected, period_min=2.0, period_max=10.0, durations=durations)
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time:.4f}s")
    print(f"CPU Result: P={cpu_res['period']:.4f}, T0={cpu_res['t0']:.4f}, Dur={cpu_res['duration']:.4f}, Depth={cpu_res['depth']:.4f}")

    # 4. Run GPU BLS
    print("\nRunning GPU BLS (CuPy Stage A1)...")
    start_time = time.time()
    gpu_res = run_gpu_bls(t, f_injected, periods, durations)
    gpu_time = time.time() - start_time
    print(f"GPU Time: {gpu_time:.4f}s")
    print(f"GPU Result: P={gpu_res['best_period']:.4f}, T0={gpu_res['best_t0']:.4f}, Dur={gpu_res['best_duration']:.4f}, Depth={gpu_res['best_depth']:.4f}")

    # 5. Compare
    p_diff = abs(cpu_res['period'] - gpu_res['best_period'])
    print(f"\nPeriod Difference: {p_diff:.6f}")
    
    if p_diff < 0.01:
        print("SUCCESS: CPU and GPU periods match closely!")
    else:
        print("FAILURE: CPU and GPU periods diverge.")

if __name__ == "__main__":
    test_accuracy()
