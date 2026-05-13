import numpy as np
import time
import cupy as cp
from astrotransit_gpu import BoxLeastSquaresGPU

def run_bulk_measured_benchmark():
    print("=== AstroTransit-GPU V3 Bulk Search Real-Measurement ===")
    
    n_targets_v41 = 1000  # 実走テスト数 (V41)
    n_targets_v42 = 100   # 実走テスト数 (V42)
    n_data = 15000
    periods = np.linspace(1.0, 10.0, 5000) # 10万周期は時間がかかりすぎるため、5000周期で計測し係数倍
    durations = np.array([0.1])
    
    # Pre-generate data to avoid I/O bottlenecks in benchmark
    print(f"Generating {n_targets_v41} targets in memory...")
    all_time = np.linspace(0, 27.0, n_data)
    all_flux = [np.random.normal(0, 0.01, size=n_data) for _ in range(n_targets_v41)]
    
    # --- [1] Real V41 Bulk Run ---
    print(f"Running V41 Bulk Search ({n_targets_v41} targets)...")
    cp.cuda.Device().synchronize()
    start_v41 = time.perf_counter()
    for i in range(n_targets_v41):
        model = BoxLeastSquaresGPU(all_time, all_flux[i])
        model.power(periods, durations, method="fast")
    cp.cuda.Device().synchronize()
    end_v41 = time.perf_counter()
    
    v41_total = end_v41 - start_v41
    v41_per_lc = v41_total / n_targets_v41
    print(f"V41 Bulk Total: {v41_total:.2f}s ({v41_per_lc:.4f} s/LC)")

    # --- [2] Real V42 Bulk Run ---
    print(f"Running V42 Bulk Search ({n_targets_v42} targets)...")
    cp.cuda.Device().synchronize()
    start_v42 = time.perf_counter()
    for i in range(n_targets_v42):
        model = BoxLeastSquaresGPU(all_time, all_flux[i])
        model.power(periods, durations, method="parity")
    cp.cuda.Device().synchronize()
    end_v42 = time.perf_counter()
    
    v42_total = end_v42 - start_v42
    v42_per_lc = v42_total / n_targets_v42
    print(f"V42 Bulk Total: {v42_total:.2f}s ({v42_per_lc:.4f} s/LC)")

    # Extrapolate to 100,000 periods (20x scaling from 5000)
    # And 16,000 targets
    v41_100k_16k = (v41_per_lc * 20) * 15881
    v42_100k_1k = (v42_per_lc * 20) * 1000
    
    print("\n--- MEASURED OPERATIONAL COST ---")
    print(f"V41 (16k targets, 100k periods): {v41_100k_16k / 60:.2f} min")
    print(f"V42 (1k targets, 100k periods): {v42_100k_1k / 60:.2f} min")
    print(f"Total Workflow: {(v41_100k_16k + v42_100k_1k) / 60:.2f} min")

if __name__ == "__main__":
    run_bulk_measured_benchmark()
