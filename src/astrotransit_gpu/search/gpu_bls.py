import cupy as cp
import numpy as np
import os

# Load CUDA kernel
KERNEL_FILE = os.path.join(os.path.dirname(__file__), "kernels", "bls.cu")
with open(KERNEL_FILE, "r") as f:
    cuda_source = f.read()

# Kernel cache for different precisions
_kernel_cache = {}

def get_kernel(dtype):
    dtype_name = np.dtype(dtype).name
    if dtype_name in _kernel_cache:
        return _kernel_cache[dtype_name]
    
    t_type = "double" if "64" in dtype_name else "float"
    options = (f"-DSCALAR_T={t_type}",)
    
    kernel = cp.RawKernel(cuda_source, "bls_kernel", options=options)
    _kernel_cache[dtype_name] = kernel
    return kernel

def run_gpu_bls(time, flux, periods, durations, flux_err=None, n_bins=200, dtype=np.float32):
    """
    Optimized GPU implementation of BLS using a custom CUDA kernel.
    Supports weighted observations and variable precision (float32/float64).
    """
    dtype = np.dtype(dtype)
    kernel = get_kernel(dtype)
    
    # 1. Check preconditions
    if len(time) != len(flux):
        raise ValueError("time and flux must have the same length")
    if flux_err is not None and len(flux_err) != len(flux):
        raise ValueError("flux_err must have the same length as flux")

    # Move data to GPU with target precision
    time_gpu = cp.asarray(time, dtype=dtype)
    flux_gpu = cp.asarray(flux, dtype=dtype)
    
    if flux_err is not None:
        err_gpu = cp.asarray(flux_err, dtype=dtype)
        weights_gpu = 1.0 / (err_gpu * err_gpu)
    else:
        weights_gpu = cp.ones_like(flux_gpu, dtype=dtype)
        
    # Subtract weighted mean
    w_sum = cp.sum(weights_gpu)
    w_mean = cp.sum(flux_gpu * weights_gpu) / w_sum
    flux_gpu -= w_mean

    inv_periods_gpu = cp.asarray(1.0 / np.asarray(periods), dtype=dtype)
    durations_gpu = cp.asarray(durations, dtype=dtype)

    n_data = len(time_gpu)
    n_periods = len(inv_periods_gpu)
    n_durations = len(durations_gpu)

    # Output buffers
    power_gpu = cp.zeros(n_periods, dtype=dtype)
    best_t0_gpu = cp.zeros(n_periods, dtype=dtype)
    best_dur_gpu = cp.zeros(n_periods, dtype=dtype)
    best_depth_gpu = cp.zeros(n_periods, dtype=dtype)

    # Kernel configuration
    N_TILE = 4
    # threads_per_block must be >= n_bins for the prefix sum and at least power of 2 for reduction
    threads_per_block = 1
    while threads_per_block < n_bins or threads_per_block < 128:
        threads_per_block *= 2
    
    if threads_per_block > 1024:
        raise ValueError(f"n_bins={n_bins} is too large for current kernel implementation (max 1024).")

    blocks_per_grid = (n_periods + N_TILE - 1) // N_TILE
    
    # Shared memory size calculation
    itemsize = dtype.itemsize
    shared_mem_size = (N_TILE * n_bins * 2 + threads_per_block * 4) * itemsize
    
    max_shm = cp.cuda.Device().attributes['MaxSharedMemoryPerBlock']
    if shared_mem_size > max_shm:
        # Fallback: Reduce N_TILE if shared memory is exceeded
        while N_TILE > 1 and shared_mem_size > max_shm:
            N_TILE //= 2
            shared_mem_size = (N_TILE * n_bins * 2 + threads_per_block * 4) * itemsize
        
        if shared_mem_size > max_shm:
            raise MemoryError(f"Shared memory limit exceeded ({shared_mem_size} > {max_shm}). Reduce n_bins.")

    # Launch kernel
    kernel(
        (blocks_per_grid,), (threads_per_block,),
        (time_gpu, flux_gpu, weights_gpu, n_data, inv_periods_gpu, n_periods,
         durations_gpu, n_durations, n_bins, dtype.type(time_gpu[0].item()),
         power_gpu, best_t0_gpu, best_dur_gpu, best_depth_gpu),
        shared_mem=shared_mem_size
    )
    # Find the best period across all searched periods
    best_p_idx = int(cp.argmax(power_gpu).item())
    
    return {
        "best_period": float(periods[best_p_idx]),
        "best_t0": float(best_t0_gpu[best_p_idx].item()),
        "best_duration": float(best_dur_gpu[best_p_idx].item()),
        "best_depth": float(best_depth_gpu[best_p_idx].item()),
        "power": power_gpu,
        "snr": float(power_gpu[best_p_idx].item()),
        "all_t0s": best_t0_gpu,
        "all_durs": best_dur_gpu,
        "all_depths": best_depth_gpu,
        "periods": np.asarray(periods, dtype=np.float32)
    }

def get_top_k_candidates(results, k=5, min_dist_bins=10):
    """
    Extract top-K independent candidates using peak finding.
    """
    from scipy.signal import find_peaks
    
    power = results['power'].get() if hasattr(results['power'], 'get') else results['power']
    peaks, _ = find_peaks(power, distance=min_dist_bins)
    
    if len(peaks) == 0:
        peaks = np.argsort(power)[-k:][::-1]

    # Sort peaks by power
    peak_powers = power[peaks]
    sorted_indices = np.argsort(peak_powers)[::-1]
    top_peaks = peaks[sorted_indices[:k]]
    
    candidates = []
    for idx in top_peaks:
        candidates.append({
            'period': float(results['periods'][idx]),
            't0': float(results['all_t0s'][idx].item()),
            'duration': float(results['all_durs'][idx].item()),
            'depth': float(results['all_depths'][idx].item()),
            'power': float(power[idx])
        })
    return candidates
