import numpy as np
import time
import cupy as cp
from astrotransit_gpu.search.api import BoxLeastSquaresGPU
from tqdm import tqdm

def run_throughput_benchmark(n_targets=100, n_data=20000, n_periods=10000):
    print(f"--- AstroTransit-GPU Survey-Scale Throughput Benchmark ---")
    print(f"Configurations:")
    print(f"  Targets: {n_targets}")
    print(f"  Data points per target: {n_data}")
    print(f"  Periods searched per target: {n_periods}")
    print(f"  Total computational load: {n_targets * n_periods:,} BLS evaluations")
    print("-" * 50)

    # Prepare search grid
    periods = np.linspace(0.5, 20.0, n_periods)
    durations = [0.05, 0.1, 0.15]
    
    # Warm up GPU
    t_warm = np.linspace(0, 10, 1000)
    y_warm = np.ones_like(t_warm)
    model_warm = BoxLeastSquaresGPU(t_warm, y_warm)
    model_warm.power(np.linspace(0.5, 5.0, 100), durations)
    cp.cuda.Stream.null.synchronize()

    # Generate synthetic data for all targets (avoiding overhead inside the loop)
    # Note: In a real pipeline, preprocessing is overlapped with GPU, 
    # here we measure pure GPU throughput.
    times = [np.linspace(0, 27.0, n_data) for _ in range(n_targets)]
    fluxes = [np.ones(n_data) + np.random.normal(0, 0.001, n_data) for _ in range(n_targets)]

    start_time = time.time()
    
    for i in tqdm(range(n_targets), desc="Processing Targets"):
        model = BoxLeastSquaresGPU(times[i], fluxes[i])
        # We don't save results to disk to measure pure compute
        _ = model.power(periods, durations, n_bins=200)

    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    targets_per_sec = n_targets / total_time
    targets_per_min = targets_per_sec * 60
    
    print("-" * 50)
    print(f"Benchmark Results:")
    print(f"  Total Time: {total_time:.2f} seconds")
    print(f"  Throughput: {targets_per_sec:.2f} targets/sec")
    print(f"  Throughput: {targets_per_min:.1f} targets/min")
    print(f"  Avg Time per Target: {total_time/n_targets*1000:.1f} ms")
    print("-" * 50)

if __name__ == "__main__":
    # You can increase n_targets to 1000 for a true stress test
    run_throughput_benchmark(n_targets=100, n_data=20000, n_periods=10000)
