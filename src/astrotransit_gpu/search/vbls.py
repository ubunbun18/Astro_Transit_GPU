import numpy as np
import os
import time
import logging
from .gpu_bls import _require_cupy

logger = logging.getLogger(__name__)

# Kernel cache
_vbls_kernel_cache = {}

def get_vbls_kernels(dtype, n_data, n_bins, n_durations, use_blackwell=None):
    cp = _require_cupy()
    dtype_name = np.dtype(dtype).name
    
    if use_blackwell is None:
        cc = cp.cuda.Device().compute_capability
        if isinstance(cc, tuple):
            major = int(cc[0])
        else:
            cc_val = int(cc)
            major = cc_val // 10 if cc_val > 20 else cc_val
        is_blackwell = int(major) >= 10
    else:
        is_blackwell = use_blackwell
    
    # V37 Apex Predator
    cache_key = f"{dtype_name}_{'v37' if is_blackwell else 'v11'}_{n_data}_{n_bins}_{n_durations}"
    if cache_key in _vbls_kernel_cache:
        return _vbls_kernel_cache[cache_key]
    
    if is_blackwell:
        kernel_file = "vbls_blackwell_v39.cu"
        kernel_name = "vbls_v39_weighted_kernel"
    else:
        kernel_file = "vbls.cu"
        kernel_name = "vbls_ultra_kernel"
    
    kernel_path = os.path.join(os.path.dirname(__file__), "kernels", kernel_file)
    with open(kernel_path, "r") as f:
        cuda_source = f.read()

    t_type = "double" if "64" in dtype_name else "float"
    
    # V26: Inject macros for maximum performance
    options = (
        f"-DSCALAR_T={t_type}",
        f"-DN_DATA={n_data}",
        f"-DN_BINS={n_bins}",
        f"-DN_DURATIONS={n_durations}"
    )
    
    mod = cp.RawModule(code=cuda_source, options=options)
    kernels = {
        "vbls_ultra": mod.get_function(kernel_name),
        "module": mod,
        "is_blackwell": is_blackwell
    }
    _vbls_kernel_cache[cache_key] = kernels
    return kernels

def run_vbls_massive(time_array, flux_matrix, periods, durations, weights_matrix=None, n_bins=128, dtype=None,
                     target_batch_size=4000, period_batch_size=20000, use_blackwell=None):
    """
    V37 Apex Predator Pipeline.
    """
    cp = _require_cupy()
    if dtype is None:
        dtype = cp.float32
    n_targets_total, n_data = flux_matrix.shape
    n_periods_total = len(periods)
    n_durations = len(durations)
    dtype = np.dtype(dtype)
    scalar_t = np.float32 if dtype == np.float32 else np.float64

    kernels = get_vbls_kernels(dtype, n_data, n_bins, n_durations, use_blackwell=use_blackwell)
    t_start = scalar_t(time_array[0].item())
    
    # V25: Ensure data is on GPU (CuPy)
    flux_matrix_gpu = cp.asarray(flux_matrix, dtype=dtype)
    time_gpu = cp.asarray(time_array, dtype=dtype)
    weights_matrix_gpu = cp.asarray(weights_matrix, dtype=dtype) if weights_matrix is not None else None
    
    dt_array_gpu = (time_gpu - t_start).astype(dtype, copy=False)
    
    dur_ptr = kernels["module"].get_global("c_durations")
    dur_gpu = cp.asarray(durations, dtype=dtype)
    cp.cuda.runtime.memcpy(dur_ptr.ptr, dur_gpu.data.ptr, n_durations * dtype.itemsize, 3)

    if kernels["is_blackwell"]:
        try:
            cp.cuda.runtime.deviceSetLimit(cp.cuda.runtime.cudaLimitPersistingL2CacheSize, 32 * 1024 * 1024)
        except:
            pass

    global_max_power = cp.zeros(n_targets_total, dtype=dtype)
    global_best_t0 = cp.zeros(n_targets_total, dtype=dtype)
    global_best_dur = cp.zeros(n_targets_total, dtype=dtype)
    global_best_depth = cp.zeros(n_targets_total, dtype=dtype)
    global_best_period = cp.zeros(n_targets_total, dtype=dtype)

    # V27: Hyper-batching for Blackwell (Defaults if not specified)
    if target_batch_size is None: target_batch_size = 8000
    if period_batch_size is None: period_batch_size = 25000
    
    from tqdm import tqdm
    
    # Progress tracking
    total_searches = n_targets_total * n_periods_total
    pbar = tqdm(total=total_searches, desc="V27 Hypernova", unit="search")
    
    # Pre-transfer fixed data
    dt_array_gpu = (time_gpu - t_start).astype(dtype, copy=False)
    dur_ptr = kernels["module"].get_global("c_durations")
    dur_gpu = cp.asarray(durations, dtype=dtype)
    cp.cuda.runtime.memcpy(dur_ptr.ptr, dur_gpu.data.ptr, n_durations * dtype.itemsize, 3)

    # Ensure weights are present (V39 requires them)
    if weights_matrix_gpu is None:
        weights_matrix_gpu = cp.ones((n_targets_total, n_data), dtype=dtype)
    
    for p_idx in range(0, n_periods_total, period_batch_size):
        p_end = min(p_idx + period_batch_size, n_periods_total)
        curr_periods = p_end - p_idx
        p_batch = periods[p_idx:p_end]
        inv_p_batch = (1.0 / p_batch).astype(dtype, copy=False)
        
        if kernels["is_blackwell"]:
            period_pairs = cp.stack((p_batch, inv_p_batch), axis=1).astype(dtype, copy=False)
        
        for t_idx in range(0, n_targets_total, target_batch_size):
            t_end = min(t_idx + target_batch_size, n_targets_total)
            curr_targets = t_end - t_idx
            batch_flux = flux_matrix_gpu[t_idx:t_end]
            batch_weights = weights_matrix_gpu[t_idx:t_end]
            
            res_pwr = global_max_power[t_idx:t_end]
            res_t0 = global_best_t0[t_idx:t_end]
            res_dur = global_best_dur[t_idx:t_end]
            res_dep = global_best_depth[t_idx:t_end]
            res_p = global_best_period[t_idx:t_end]

            grid = (int((curr_periods + 15) // 16), int(curr_targets))

            if kernels["is_blackwell"]:
                # V39 Dynamic SMEM Calculation (3 * N_DATA for flux, dt, weight)
                smem_size = (3 * n_data * dtype.itemsize) + (2 * 16 * (n_bins + 1) * dtype.itemsize)
                kernels["vbls_ultra"].max_dynamic_shared_size_bytes = smem_size
                
                kernels["vbls_ultra"](
                    grid, (512,),
                    (batch_flux, dt_array_gpu, period_pairs,
                     batch_weights,
                     res_pwr, res_t0, res_dur, res_dep, res_p,
                     curr_periods, t_start),
                    shared_mem=smem_size
                )
            else:
                params_i32 = cp.asarray([n_data, curr_periods, n_durations, n_bins], dtype=cp.int32)
                params_scalar = cp.asarray([t_start], dtype=dtype)
                kernels["vbls_ultra"](
                    grid, (512,),
                    (batch_flux, time_gpu, inv_p_batch, p_batch,
                     res_pwr, res_t0, res_dur, res_dep, res_p,
                     params_i32, params_scalar)
                )
            
            # V27: Synchronize every 8 batches to get accurate progress
            if t_idx % (target_batch_size * 8) == 0:
                 cp.cuda.Stream.null.synchronize()
                 # Real-time G-searches/sec estimation
                 elapsed = time.time() - pbar.start_t
                 if elapsed > 0:
                     g_searches = pbar.n / elapsed / 1e9
                     pbar.set_postfix({"G-searches/s": f"{g_searches:.2f}"})
            
            pbar.update(curr_targets * curr_periods)

    cp.cuda.Stream.null.synchronize()
    pbar.close()

    return {
        "best_period": global_best_period,
        "best_t0": global_best_t0,
        "best_depth": global_best_depth,
        "best_duration": global_best_dur,
        "snr": global_max_power
    }
