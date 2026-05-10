import numpy as np
import os

# Kernel cache for different precisions
_kernel_cache = {}

def _require_cupy():
    """Lazy import of CuPy to allow CPU-only environments to import the package."""
    try:
        import cupy as cp
        return cp
    except ImportError as e:
        raise ImportError(
            "CuPy is required for GPU acceleration. "
            "Install with: pip install 'astrotransit-gpu[cuda12]'"
        ) from e

def get_kernel(dtype, n_tile=4):
    cp = _require_cupy()
    
    # Load CUDA kernel
    kernel_path = os.path.join(os.path.dirname(__file__), "kernels", "bls.cu")
    with open(kernel_path, "r") as f:
        cuda_source = f.read()

    dtype_name = np.dtype(dtype).name
    # Cache key includes n_tile to avoid mismatch
    cache_key = (dtype_name, n_tile)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]
    
    t_type = "double" if "64" in dtype_name else "float"
    options = (
        f"-DSCALAR_T={t_type}",
        f"-DN_TILE={n_tile}",
    )
    
    kernel = cp.RawKernel(cuda_source, "bls_kernel", options=options)
    _kernel_cache[cache_key] = kernel
    return kernel

def run_gpu_bls(time, flux, periods, durations, flux_err=None, n_bins=200, dtype=np.float32):
    """
    Optimized GPU implementation of BLS using a custom CUDA kernel.
    Supports weighted observations and variable precision (float32/float64).
    """
    cp = _require_cupy()
    dtype = np.dtype(dtype)
    
    # 1. Check preconditions
    if len(time) != len(flux):
        raise ValueError("time and flux must have the same length")
    if flux_err is not None and len(flux_err) != len(flux):
        raise ValueError("flux_err must have the same length as flux")

    # Move data to GPU if needed
    time_gpu = cp.asarray(time, dtype=dtype)
    flux_gpu = cp.asarray(flux, dtype=dtype).copy() # Copy to avoid modifying input if we subtract mean
    
    if flux_err is not None:
        err_gpu = cp.asarray(flux_err, dtype=dtype)
        weights_gpu = 1.0 / (err_gpu * err_gpu)
    else:
        weights_gpu = cp.ones_like(flux_gpu, dtype=dtype)
        
    # Subtract weighted mean
    w_sum = cp.sum(weights_gpu)
    w_mean = cp.sum(flux_gpu * weights_gpu) / w_sum
    flux_gpu -= w_mean

    # Use pre-computed GPU arrays if provided, else transfer
    if isinstance(periods, cp.ndarray):
        inv_periods_gpu = 1.0 / periods.astype(dtype, copy=False)
        periods_cpu = periods.get()
    else:
        periods_np = np.asarray(periods)
        inv_periods_gpu = cp.asarray(1.0 / periods_np, dtype=dtype)
        periods_cpu = periods_np

    if isinstance(durations, cp.ndarray):
        durations_gpu = durations.astype(dtype, copy=False)
    else:
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
    n_tile = 4
    threads_per_block = 1
    while threads_per_block < n_bins or threads_per_block < 128:
        threads_per_block *= 2
    
    if threads_per_block > 1024:
        raise ValueError(f"n_bins={n_bins} is too large for current kernel implementation (max 1024).")

    # Shared memory check and dynamic tile reduction
    itemsize = dtype.itemsize
    shared_mem_size = (n_tile * n_bins * 2 + threads_per_block * 4) * itemsize
    max_shm = cp.cuda.Device().attributes['MaxSharedMemoryPerBlock']
    
    while n_tile > 1 and shared_mem_size > max_shm:
        n_tile //= 2
        shared_mem_size = (n_tile * n_bins * 2 + threads_per_block * 4) * itemsize
        
    if shared_mem_size > max_shm:
        raise MemoryError(f"Shared memory limit exceeded ({shared_mem_size} > {max_shm}). Reduce n_bins.")

    # Get/compile kernel with decided n_tile
    kernel = get_kernel(dtype, n_tile=n_tile)
    blocks_per_grid = (n_periods + n_tile - 1) // n_tile

    # Launch kernel
    t_start = float(time_gpu[0]) if n_data > 0 else 0.0
    kernel(
        (blocks_per_grid,), (threads_per_block,),
        (time_gpu, flux_gpu, weights_gpu, n_data, inv_periods_gpu, n_periods,
         durations_gpu, n_durations, n_bins, dtype.type(t_start),
         power_gpu, best_t0_gpu, best_dur_gpu, best_depth_gpu),
        shared_mem=shared_mem_size
    )

    # Find the best period
    best_p_idx = int(cp.argmax(power_gpu).item())
    
    return {
        "best_period": float(periods_cpu[best_p_idx]),
        "best_t0": float(best_t0_gpu[best_p_idx].item()),
        "best_duration": float(best_dur_gpu[best_p_idx].item()),
        "best_depth": float(best_depth_gpu[best_p_idx].item()),
        "power": power_gpu,
        "snr": float(power_gpu[best_p_idx].item()),
        "all_t0s": best_t0_gpu,
        "all_durs": best_dur_gpu,
        "all_depths": best_depth_gpu,
        "periods": periods_cpu
    }

def get_top_k_candidates(results, k=5, min_dist_bins=10):
    from scipy.signal import find_peaks
    
    # Move all potential GPU arrays to CPU once
    power = results['power'].get() if hasattr(results['power'], 'get') else results['power']
    all_t0s = results['all_t0s'].get() if hasattr(results['all_t0s'], 'get') else results['all_t0s']
    all_durs = results['all_durs'].get() if hasattr(results['all_durs'], 'get') else results['all_durs']
    all_depths = results['all_depths'].get() if hasattr(results['all_depths'], 'get') else results['all_depths']
    periods = results['periods'] # Already on CPU
    
    peaks, _ = find_peaks(power, distance=min_dist_bins)
    
    if len(peaks) == 0:
        peaks = np.argsort(power)[-k:][::-1]

    peak_powers = power[peaks]
    sorted_indices = np.argsort(peak_powers)[::-1]
    top_peaks = peaks[sorted_indices[:k]]
    
    candidates = []
    for idx in top_peaks:
        candidates.append({
            'period': float(periods[idx]),
            't0': float(all_t0s[idx]),
            'duration': float(all_durs[idx]),
            'depth': float(all_depths[idx]),
            'power': float(power[idx])
        })
    return candidates
