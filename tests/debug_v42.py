import numpy as np
import cupy as cp
from astropy.timeseries import BoxLeastSquares
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from astrotransit_gpu.search.v42_parity import run_vbls_exact_parity

def debug_v42_parity():
    np.random.seed(42)
    time = np.linspace(0, 27, 200).astype(np.float64)
    flux = np.random.normal(0, 0.003, size=200).astype(np.float64)
    
    true_period = 3.521
    true_dur = 0.1
    ph = (time % true_period) / true_period
    flux[ph < (true_dur / true_period)] -= 0.01
    flux -= np.median(flux)
    
    dy = np.ones_like(flux) * 0.003

    periods = np.linspace(3.0, 4.0, 1000).astype(np.float64)
    durations = np.array([0.05, 0.1, 0.15]).astype(np.float64)
    
    # 1. CPU Astropy
    model_cpu = BoxLeastSquares(time, flux, dy=dy)
    res_cpu = model_cpu.power(periods, durations, objective="snr", oversample=10)
    cpu_power = res_cpu.power
    
    # 2. GPU V42
    weights = 1.0 / (dy**2)
    flux_median_subtracted = flux - np.median(flux)
    res_gpu = run_vbls_exact_parity(
        time, flux_median_subtracted.reshape(1, -1),
        periods, durations,
        weights_matrix=weights.reshape(1, -1),
        oversample=10,
        max_bins=4000,
        dtype=cp.float64
    )
    
    gpu_power = cp.asnumpy(res_gpu["power_array"][0])
    gpu_dur = cp.asnumpy(res_gpu["dur_array"][0])
    gpu_t0 = cp.asnumpy(res_gpu["t0_array"][0])
    
    print(f"Astropy Power [:10]: {cpu_power[:10]}")
    print(f"GPU Power     [:10]: {gpu_power[:10]}")
    
    diff = np.abs(cpu_power - gpu_power)
    max_idx = np.argmax(diff)
    print(f"Max Diff at index {max_idx}: Astropy={cpu_power[max_idx]}, GPU={gpu_power[max_idx]}")
    print(f"Period at max diff: {periods[max_idx]}")
    print(f"Astropy T0/Dur: {res_cpu.transit_time[max_idx]}, {res_cpu.duration[max_idx]}")
    print(f"GPU T0/Dur:     {gpu_t0[max_idx]}, {gpu_dur[max_idx]}")
    
    # Also print correlation
    correlation = np.corrcoef(cpu_power, gpu_power)[0, 1]
    print(f"Correlation: {correlation}")
    
if __name__ == "__main__":
    debug_v42_parity()
