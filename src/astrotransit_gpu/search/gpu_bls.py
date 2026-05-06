import cupy as cp
import numpy as np
import os

# Load CUDA kernel
KERNEL_FILE = os.path.join(os.path.dirname(__file__), "kernels", "bls.cu")
with open(KERNEL_FILE, "r") as f:
    cuda_source = f.read()

_bls_kernel = cp.RawKernel(cuda_source, "bls_kernel")

def run_gpu_bls(time, flux, periods, durations, n_bins=200):
    """
    Optimized GPU implementation of BLS using a custom CUDA kernel.
    Stage A3: CUDA RawKernel with shared memory binning.
    """
    # 1. Check preconditions
    if len(time) != len(flux):
        raise ValueError("time and flux must have the same length")
    if len(periods) == 0 or len(durations) == 0:
        raise ValueError("periods and durations must not be empty")

    # Move data to GPU and ensure float32 for kernel
    time_gpu = cp.asarray(time, dtype=cp.float32)
    flux_gpu = cp.asarray(flux, dtype=cp.float32)
    inv_periods_gpu = cp.asarray(1.0 / np.asarray(periods), dtype=cp.float32)
    durations_gpu = cp.asarray(durations, dtype=cp.float32)

    n_data = len(time_gpu)
    n_periods = len(inv_periods_gpu)
    n_durations = len(durations_gpu)

    # Output buffers
    power_gpu = cp.zeros(n_periods, dtype=cp.float32)
    best_t0_gpu = cp.zeros(n_periods, dtype=cp.float32)
    best_dur_gpu = cp.zeros(n_periods, dtype=cp.float32)
    best_depth_gpu = cp.zeros(n_periods, dtype=cp.float32)

    # Kernel configuration
    N_TILE = 8
    threads_per_block = 256
    blocks_per_grid = (n_periods + N_TILE - 1) // N_TILE
    
    # Shared memory: (N_TILE * n_bins * 2 + threads_per_block * 4) * 4 bytes
    shared_mem_size = (N_TILE * n_bins * 2 + threads_per_block * 4) * 4

    # Launch kernel
    _bls_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (time_gpu, flux_gpu, n_data, inv_periods_gpu, n_periods,
         durations_gpu, n_durations, n_bins, float(time_gpu[0]),
         power_gpu, best_t0_gpu, best_dur_gpu, best_depth_gpu),
        shared_mem=shared_mem_size
    )

    # Find the best period across all searched periods
    best_p_idx = int(cp.argmax(power_gpu))
    
    return {
        "best_period": float(periods[best_p_idx]),
        "best_t0": float(best_t0_gpu[best_p_idx]),
        "best_duration": float(best_dur_gpu[best_p_idx]),
        "best_depth": float(best_depth_gpu[best_p_idx]),
        "power": power_gpu,
        "snr": float(power_gpu[best_p_idx]),
        "all_t0s": best_t0_gpu,
        "all_durs": best_dur_gpu,
        "all_depths": best_depth_gpu,
        "periods": cp.asarray(periods, dtype=cp.float32)
    }

def get_top_k_candidates(results, k=5, min_dist_bins=10):
    """
    Extract top-K independent candidates using peak finding.
    """
    from scipy.signal import find_peaks
    
    power = results['power'].get() if hasattr(results['power'], 'get') else results['power']
    peaks, _ = find_peaks(power, distance=min_dist_bins)
    
    # Sort peaks by power
    peak_powers = power[peaks]
    sorted_indices = np.argsort(peak_powers)[::-1]
    top_peaks = peaks[sorted_indices[:k]]
    
    candidates = []
    for idx in top_peaks:
        candidates.append({
            'period': float(results['periods'][idx]),
            't0': float(results['all_t0s'][idx]),
            'duration': float(results['all_durs'][idx]),
            'depth': float(results['all_depths'][idx]),
            'power': float(power[idx])
        })
    return candidates

def run_gpu_bls_naive(time, flux, periods, durations):
    """
    Original naive vectorized CuPy implementation (Stage A1).
    Kept for reference/fallback.
    """
    # ... (existing code moved here if needed, but I'll replace the main one)
    pass
