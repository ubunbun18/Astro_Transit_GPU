import numpy as np
import time
import cupy as cp
from astrotransit_gpu import BoxLeastSquaresGPU
from astrotransit_gpu.search.cpu_reference_bls import run_astropy_bls

def run_v3_benchmark():
    print("=== AstroTransit-GPU V3 Benchmark & Integrity Validation ===")
    
    # 1. Setup Sample Data
    n_data = 15000  # Typical TESS LC size
    time_arr = np.linspace(0, 27.0, n_data)
    flux = np.random.normal(0, 0.01, size=n_data)
    # Inject a realistic transit
    period_true = 3.5218
    t0_true = 1.25
    duration_true = 0.12
    depth_true = 0.005
    mask = (time_arr % period_true < duration_true/2) | (time_arr % period_true > period_true - duration_true/2)
    flux[mask] -= depth_true
    
    periods = np.linspace(1.0, 10.0, 5000)
    durations = np.array([0.05, 0.1, 0.15, 0.2])
    
    model = BoxLeastSquaresGPU(time_arr, flux)
    
    # --- CPU Reference ---
    print("\n[1/3] Running CPU Reference (Astropy)...")
    s = time.time()
    cpu_res = run_astropy_bls(time_arr, flux, periods=periods, durations=durations)
    cpu_time = time.time() - s
    print(f"CPU Time: {cpu_time:.4f}s")

    # --- V41 (Fast Mode) ---
    print("\n[2/3] Validating V41 (Fast Mode)...")
    v41_times = []
    for _ in range(5):
        s = time.time()
        v41_res = model.power(periods, durations, method="fast", dtype=np.float32)
        v41_times.append(time.time() - s)
    
    v41_med = np.median(v41_times)
    v41_corr = np.corrcoef(cpu_res['power'], v41_res.power)[0, 1]
    v41_rmse = np.sqrt(np.mean((cpu_res['power'] - v41_res.power)**2))
    
    print(f"V41 Throughput: {1.0/v41_med:.2f} LC/s")
    print(f"V41 Correlation: {v41_corr:.10f}")
    print(f"V41 RMSE: {v41_rmse:.2e}")

    # --- V42 (Parity Mode) ---
    print("\n[3/3] Validating V42 (Parity Mode)...")
    v42_times = []
    for _ in range(3): # Slower, so fewer runs
        s = time.time()
        v42_res = model.power(periods, durations, method="parity", oversample=10)
        v42_times.append(time.time() - s)
    
    v42_med = np.median(v42_times)
    v42_corr = np.corrcoef(cpu_res['power'], v42_res.power)[0, 1]
    v42_rmse = np.sqrt(np.mean((cpu_res['power'] - v42_res.power)**2))
    
    print(f"V42 Throughput: {1.0/v42_med:.2f} LC/s")
    print(f"V42 Correlation: {v42_corr:.10f}")
    print(f"V42 RMSE: {v42_rmse:.2e}")

    # Final Summary Table for Report
    print("\n=== FINAL METRICS ===")
    print(f"| Kernel | LC/s | Correlation | RMSE | Speedup |")
    print(f"| :--- | :--- | :--- | :--- | :--- |")
    print(f"| CPU (Astropy) | {1.0/cpu_time:.4f} | 1.0000000000 | 0.00e+00 | 1x |")
    print(f"| V41 (Fast) | {1.0/v41_med:.2f} | {v41_corr:.10f} | {v41_rmse:.2e} | {cpu_time/v41_med:.1f}x |")
    print(f"| V42 (Parity) | {1.0/v42_med:.2f} | {v42_corr:.10f} | {v42_rmse:.2e} | {cpu_time/v42_med:.1f}x |")

if __name__ == "__main__":
    run_v3_benchmark()
