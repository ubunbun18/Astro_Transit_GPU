import numpy as np
import cupy as cp
import time
import sys
import os
import logging
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from astrotransit_gpu.search.vbls import run_vbls_massive

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_single_test(n_targets, n_periods, n_data, t_batch, p_batch, dtype=np.float32):
    # Generate dummy data
    time_array = np.linspace(0, 27, n_data).astype(dtype)
    flux_matrix = np.random.normal(0, 0.001, size=(n_targets, n_data)).astype(dtype)
    periods = np.linspace(0.5, 13.0, n_periods).astype(dtype)
    durations = np.linspace(0.01, 0.2, 8).astype(dtype)
    
    # Warm-up (only once per script start)
    # cp.cuda.runtime.deviceSynchronize()
    
    start_time = time.time()
    results = run_vbls_massive(
        time_array, flux_matrix, cp.asarray(periods), durations,
        target_batch_size=t_batch,
        period_batch_size=p_batch,
        dtype=dtype
    )
    cp.cuda.runtime.deviceSynchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = n_targets / total_time
    # GCPS = (Targets * Periods * Durations) / Time
    gcps = (n_targets * n_periods * len(durations)) / total_time / 1e9
    
    return {
        "n_targets": n_targets,
        "n_periods": n_periods,
        "n_data": n_data,
        "t_batch": t_batch,
        "p_batch": p_batch,
        "total_time": total_time,
        "throughput": throughput,
        "gcps": gcps
    }

def main():
    logger.info("--- Starting Comprehensive Performance Benchmark ---")
    
    results = []
    
    # 1. Grid Density Scaling (Fixed targets=4000, batches=Standard)
    logger.info("Scaling Grid Density...")
    for n_p in [10000, 100000, 500000]:
        res = run_single_test(4000, n_p, 1312, 8000, 25000)
        results.append(res)
        logger.info(f"N_Periods={n_p}: {res['throughput']:.2f} LC/s, {res['gcps']:.2f} GCPS")

    # 2. Batch Size Optimization (Fixed targets=8000, periods=100000)
    logger.info("Optimizing Batch Sizes...")
    for t_b in [128, 1024, 4096, 8000]:
        for p_b in [10000, 25000, 50000]:
            res = run_single_test(8000, 100000, 1312, t_b, p_b)
            results.append(res)
            logger.info(f"T_Batch={t_b}, P_Batch={p_b}: {res['throughput']:.2f} LC/s")

    # 3. Data Density Scaling
    logger.info("Scaling Data Density...")
    for n_d in [1312, 10000]:
        res = run_single_test(2000, 100000, n_d, 8000, 25000)
        results.append(res)
        logger.info(f"N_Data={n_d}: {res['throughput']:.2f} LC/s")

    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join("reports", "comprehensive_performance_results.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    # Final Summary
    best_throughput = df["throughput"].max()
    best_gcps = df["gcps"].max()
    logger.info(f"--- Benchmark Complete ---")
    logger.info(f"Peak Throughput: {best_throughput:.2f} LC/s")
    logger.info(f"Peak GCPS: {best_gcps:.2f} GCPS")

if __name__ == "__main__":
    main()
