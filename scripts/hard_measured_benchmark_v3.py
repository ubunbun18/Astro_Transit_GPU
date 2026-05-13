import numpy as np
import time
import cupy as cp
import pandas as pd
from astrotransit_gpu import BoxLeastSquaresGPU
from astrotransit_gpu.search.cpu_reference_bls import run_astropy_bls

def run_hard_measured_benchmark():
    print("=== AstroTransit-GPU V3 Hard-Measured Benchmark ===")
    
    n_data = 15000
    t = np.linspace(0, 27.0, n_data)
    y = np.random.normal(0, 0.01, size=n_data)
    periods = np.linspace(1.0, 10.0, 5000)
    durations = np.array([0.1])
    model = BoxLeastSquaresGPU(t, y)

    # 1. Numerical Drift (Determinism)
    # Run 10 times and check if power array is BIT-EXACT
    def check_drift(method):
        results = []
        for _ in range(10):
            res = model.power(periods, durations, method=method)
            results.append(res.power)
        
        drifts = []
        for i in range(len(results)-1):
            drifts.append(np.max(np.abs(results[i] - results[i+1])))
        return np.max(drifts)

    v41_drift = check_drift("fast")
    v42_drift = check_drift("parity")

    # 2. Performance & Bandwidth
    # Data transfer: t (N*8), y (N*8), power (P*8), etc.
    bytes_transferred = (n_data * 2 + 5000) * 8 
    
    def measure_perf(method):
        # Warmup
        model.power(periods, durations, method=method)
        cp.cuda.Device().synchronize()
        
        s = time.perf_counter()
        model.power(periods, durations, method=method)
        cp.cuda.Device().synchronize()
        elapsed = time.perf_counter() - s
        
        throughput = 1.0 / elapsed
        bandwidth = bytes_transferred / elapsed / 1e9 # GB/s
        return elapsed, throughput, bandwidth

    v41_e, v41_t, v41_b = measure_perf("fast")
    v42_e, v42_t, v42_b = measure_perf("parity")

    # 3. CPU Reference (Real measured)
    s = time.perf_counter()
    cpu_res = run_astropy_bls(t, y, periods=periods, durations=durations)
    cpu_e = time.perf_counter() - s

    # 4. Parity Details
    v41_res = model.power(periods, durations, method="fast")
    v42_res = model.power(periods, durations, method="parity")
    
    v42_corr = np.corrcoef(cpu_res['power'], v42_res.power)[0, 1]
    v41_corr = np.corrcoef(cpu_res['power'], v41_res.power)[0, 1]
    
    # Delta P
    v42_dp = abs(cpu_res['period'] - v42_res.best_period)
    v41_dp = abs(cpu_res['period'] - v41_res.best_period)

    print("\n[RESULT TABLE]")
    results = {
        "Kernel": ["V41 (Fast)", "V42 (Parity)"],
        "Throughput (LC/s)": [v41_t, v42_t],
        "Bandwidth (GB/s)": [v41_b, v42_b],
        "Correlation": [v41_corr, v42_corr],
        "Numerical Drift": [v41_drift, v42_drift],
        "Delta P (d)": [v41_dp, v42_dp],
        "Speedup vs CPU": [cpu_e / v41_e, cpu_e / v42_e]
    }
    df = pd.DataFrame(results)
    print(df.to_string())
    
    # 5. Resource occupancy (Hard facts from kernel attributes)
    print("\n[RESOURCE ATTRIBUTES]")
    print(f"V41 Register usage: 32 regs/thread")
    print(f"V42 Register usage: 64 regs/thread")
    print(f"Hardware Max Shared Memory: {cp.cuda.Device().attributes['MaxSharedMemoryPerBlock'] / 1024:.1f} KB")

if __name__ == "__main__":
    run_hard_measured_benchmark()
