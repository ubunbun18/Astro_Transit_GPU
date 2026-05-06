import numpy as np
import cupy as cp
import time
from astrotransit_gpu.search.gpu_bls import run_gpu_bls
from astrotransit_gpu.inject.box import inject_box_transit

def run_performance_bench():
    # 1. Create synthetic data
    t = np.linspace(0, 20, 10000)
    f = np.ones_like(t)
    f = inject_box_transit(t, f, 5.321, 1.2, 0.12, 0.005)
    f += np.random.normal(0, 0.0005, size=len(t))
    
    # 2. Define large search grid
    n_periods = 10000
    periods = np.linspace(0.5, 20.0, n_periods)
    durations = np.linspace(0.01, 0.2, 10)
    
    print(f"Running GPU BLS Benchmark with {n_periods} periods...")
    
    # Warmup
    _ = run_gpu_bls(t, f, periods[:10], durations)
    
    start_time = time.time()
    results = run_gpu_bls(t, f, periods, durations)
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"GPU Time for {n_periods} periods: {elapsed:.4f}s")
    print(f"Throughput: {n_periods / elapsed:.2f} periods/sec")
    print(f"Best Period: {results['best_period']:.6f}")

if __name__ == "__main__":
    run_performance_bench()
