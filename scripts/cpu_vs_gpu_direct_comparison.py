import numpy as np
import cupy as cp
import time
import sys
import os
import logging
from astropy.timeseries import BoxLeastSquares

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from astrotransit_gpu.search.vbls import run_vbls_massive

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    n_targets = 10
    n_periods = 100000
    n_data = 1312
    durations = np.linspace(0.01, 0.2, 5)
    periods = np.linspace(0.5, 13.0, n_periods)
    
    logger.info(f"--- Direct CPU vs GPU Comparison ---")
    logger.info(f"Targets: {n_targets}, Periods: {n_periods}, Data Points: {n_data}")
    
    # Data generation
    t = np.linspace(0, 27, n_data)
    y_matrix = np.random.normal(0, 0.01, size=(n_targets, n_data)).astype(np.float32)
    dy = 0.01
    
    # 1. CPU Measurement (Astropy)
    logger.info("Starting CPU (Astropy) benchmark...")
    cpu_start = time.time()
    for i in range(n_targets):
        model = BoxLeastSquares(t, y_matrix[i])
        # Note: durations in Astropy is used for search grid
        res = model.power(periods, durations)
    cpu_end = time.time()
    cpu_time = cpu_end - cpu_start
    cpu_throughput = n_targets / cpu_time
    logger.info(f"CPU Time: {cpu_time:.4f} sec ({cpu_throughput:.4f} LC/s)")
    
    # 2. GPU Measurement (V39)
    logger.info("Starting GPU (V39 Blackwell) benchmark...")
    # Warm-up
    _ = run_vbls_massive(t, y_matrix[:1], cp.asarray(periods[:100]), durations, dtype=np.float32)
    cp.cuda.runtime.deviceSynchronize()
    
    gpu_start = time.time()
    _ = run_vbls_massive(t, y_matrix, cp.asarray(periods), durations, dtype=np.float32)
    cp.cuda.runtime.deviceSynchronize()
    gpu_end = time.time()
    gpu_time = gpu_end - gpu_start
    gpu_throughput = n_targets / gpu_time
    logger.info(f"GPU Time: {gpu_time:.4f} sec ({gpu_throughput:.4f} LC/s)")
    
    # 3. Results
    speedup = cpu_time / gpu_time
    logger.info(f"--- Result Summary ---")
    logger.info(f"Speedup Factor: {speedup:.2f}x")
    logger.info(f"GPU is {speedup:.2f} times faster than single-core CPU implementation.")

if __name__ == "__main__":
    main()
