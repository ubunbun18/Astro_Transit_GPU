import numpy as np
import cupy as cp
import time
import pandas as pd
from astrotransit_gpu.search.gpu_bls import run_gpu_bls
from astrotransit_gpu.inject.box import inject_box_transit

def run_extreme_bench():
    # 1. Prepare Large Data (100,000 points)
    n_data = 100000
    t = np.linspace(0, 50, n_data)
    f = np.ones_like(t)
    f = inject_box_transit(t, f, 12.345, 1.5, 0.2, 0.01)
    f += np.random.normal(0, 0.001, size=n_data)
    
    # 2. Define Massive Grid (100,000 periods)
    n_periods = 100000
    periods = np.linspace(1.0, 40.0, n_periods)
    durations = np.linspace(0.05, 0.3, 5)
    
    print(f"--- Extreme Scale Benchmark ---")
    print(f"Data Points: {n_data}")
    print(f"Period Grid: {n_periods} points")
    print(f"Total Trials: {n_periods * len(durations):,} (Period x Duration combinations)")
    
    # GPU Execution
    print("\nStarting GPU BLS...")
    cp.cuda.Device(0).synchronize()
    start_gpu = time.time()
    res_gpu = run_gpu_bls(t, f, periods, durations)
    cp.cuda.Device(0).synchronize()
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu
    
    # CPU Execution (Subsampled for safety)
    print("Estimating CPU BLS time (based on 100 periods)...")
    from astropy.timeseries import BoxLeastSquares
    model = BoxLeastSquares(t, f)
    start_cpu = time.time()
    _ = model.power(periods[:100], durations)
    end_cpu = time.time()
    cpu_est_time = (end_cpu - start_cpu) * (n_periods / 100)
    
    print("\n--- RESULTS ---")
    print(f"GPU Runtime: {gpu_time:.4f} s")
    print(f"CPU Estimated Runtime: {cpu_est_time:.2f} s")
    print(f"Speedup: {cpu_est_time / gpu_time:.1f}x faster")
    print(f"Throughput: {n_periods / gpu_time:,.0f} periods/sec")
    
    # Results Check
    print(f"\nDetected Period: {res_gpu['best_period']:.6f} (Target: 12.345)")

if __name__ == "__main__":
    run_extreme_bench()
