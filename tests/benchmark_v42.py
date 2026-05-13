import numpy as np
import cupy as cp
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from astrotransit_gpu.search.vbls import run_vbls_massive
from astrotransit_gpu.search.v42_parity import run_vbls_exact_parity

def benchmark_v41_vs_v42():
    # データの準備
    N_TARGETS = 100
    N_DATA = 1312
    N_PERIODS = 100000
    
    time_arr = np.linspace(0, 27, N_DATA).astype(np.float32)
    flux_batch = np.random.normal(0, 0.003, (N_TARGETS, N_DATA)).astype(np.float32)
    periods = np.linspace(0.5, 13.0, N_PERIODS).astype(np.float32)
    durations = np.linspace(0.01, 0.2, 8).astype(np.float32)
    
    print(f"Benchmarking {N_TARGETS} LCs x {N_PERIODS:,} periods...")

    # --- V41 (Fast Mode) ---
    # Warmup
    _ = run_vbls_massive(time_arr, flux_batch[:2], cp.asarray(periods), durations)
    cp.cuda.runtime.deviceSynchronize()
    
    t0 = time.perf_counter()
    _ = run_vbls_massive(time_arr, flux_batch, cp.asarray(periods), durations)
    cp.cuda.runtime.deviceSynchronize()
    v41_time = time.perf_counter() - t0
    v41_throughput = N_TARGETS / v41_time

    # --- V42 (Parity Mode) ---
    # Warmup
    _ = run_vbls_exact_parity(time_arr, flux_batch[:2], periods, durations)
    cp.cuda.runtime.deviceSynchronize()
    
    t0 = time.perf_counter()
    _ = run_vbls_exact_parity(time_arr, flux_batch, periods, durations)
    cp.cuda.runtime.deviceSynchronize()
    v42_time = time.perf_counter() - t0
    v42_throughput = N_TARGETS / v42_time

    print(f"\nResults:")
    print(f"V41 (Fast, 128-bins):    {v41_throughput:>10.2f} LC/s")
    print(f"V42 (Parity, Dynamic):  {v42_throughput:>10.2f} LC/s")
    print(f"Overhead:               {((v41_throughput/v42_throughput)-1)*100:.1f}%")

if __name__ == "__main__":
    benchmark_v41_vs_v42()
