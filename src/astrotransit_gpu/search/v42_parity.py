import numpy as np
import os
from .gpu_bls import _require_cupy

_v42_kernel_cache = {}

def get_v42_kernel(dtype):
    cp = _require_cupy()
    dtype_name = np.dtype(dtype).name
    if dtype_name in _v42_kernel_cache:
        return _v42_kernel_cache[dtype_name]

    kernel_path = os.path.join(os.path.dirname(__file__), "kernels", "vbls_v42_parity.cu")
    with open(kernel_path, "r") as f:
        cuda_source = f.read()

    t_type = "double" if "64" in dtype_name else "float"
    options = (
        f"-DSCALAR_T={t_type}",
    )
    
    mod = cp.RawModule(code=cuda_source, options=options)
    kernel = mod.get_function("vbls_v42_parity_kernel")
    _v42_kernel_cache[dtype_name] = kernel
    return kernel

def run_vbls_exact_parity(time_array, flux_matrix, periods, durations, weights_matrix=None,
                          oversample=10, max_bins=2000, dtype=None):
    """
    V42 Exact Parity Kernel - Returns exactly Astropy objective="snr" values.
    """
    cp = _require_cupy()
    if dtype is None:
        dtype = cp.float32
    n_targets, n_data = flux_matrix.shape
    n_periods = len(periods)
    n_durations = len(durations)
    
    dtype = np.dtype(dtype)
    scalar_t = np.float32 if dtype == np.float32 else np.float64
    
    t_start = scalar_t(time_array[0].item())
    
    flux_matrix_gpu = cp.asarray(flux_matrix, dtype=dtype)
    time_gpu = cp.asarray(time_array, dtype=dtype)
    dt_array_gpu = (time_gpu - t_start).astype(dtype, copy=False)
    
    if weights_matrix is None:
        weights_matrix_gpu = cp.ones((n_targets, n_data), dtype=dtype)
    else:
        weights_matrix_gpu = cp.asarray(weights_matrix, dtype=dtype)
        
    periods_gpu = cp.asarray(periods, dtype=dtype)
    durations_gpu = cp.asarray(durations, dtype=dtype)
    
    out_power = cp.zeros(n_targets * n_periods, dtype=dtype)
    out_t0 = cp.zeros(n_targets * n_periods, dtype=dtype)
    out_dur = cp.zeros(n_targets * n_periods, dtype=dtype)
    out_dep = cp.zeros(n_targets * n_periods, dtype=dtype)
    
    kernel = get_v42_kernel(dtype)
    
    grid = (n_periods, n_targets)
    block = (256, 1, 1)
    
    sum_y_total = cp.sum(flux_matrix_gpu * weights_matrix_gpu, axis=1).astype(dtype)
    sum_ivar_total = cp.sum(weights_matrix_gpu, axis=1).astype(dtype)
    
    # 1-Target per block. Max_bins=2000 is ~32KB.
    smem_size = (max_bins + 150) * dtype.itemsize * 2
    
    # Explicitly set shared memory limit for the function
    try:
        # Some CuPy versions require setting it this way
        kernel.max_dynamic_shared_size_bytes = int(smem_size)
    except Exception:
        pass
        
    kernel(
        grid, block,
        (flux_matrix_gpu, dt_array_gpu, periods_gpu, durations_gpu, weights_matrix_gpu,
         out_power, out_t0, out_dur, out_dep,
         np.int32(n_data), np.int32(n_durations), np.int32(oversample), np.int32(max_bins), 
         scalar_t(t_start),
         sum_y_total, sum_ivar_total, np.int32(n_targets)),
        shared_mem=int(smem_size)
    )
    
    out_power = out_power.reshape(n_targets, n_periods)
    out_t0 = out_t0.reshape(n_targets, n_periods)
    out_dur = out_dur.reshape(n_targets, n_periods)
    out_dep = out_dep.reshape(n_targets, n_periods)
    
    best_idx = cp.argmax(out_power, axis=1)
    target_indices = cp.arange(n_targets)
    
    return {
        "snr": out_power[target_indices, best_idx],
        "best_period": periods_gpu[best_idx],
        "best_t0": out_t0[target_indices, best_idx],
        "best_duration": out_dur[target_indices, best_idx],
        "best_depth": out_dep[target_indices, best_idx],
        "power_array": out_power,
        "t0_array": out_t0,
        "dur_array": out_dur,
        "depth_array": out_dep
    }
