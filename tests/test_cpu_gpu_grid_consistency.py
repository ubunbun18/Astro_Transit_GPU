import numpy as np
import pytest
from astrotransit_gpu.search.gpu_bls import run_gpu_bls
from astrotransit_gpu.search.cpu_reference_bls import run_astropy_bls
from astrotransit_gpu.inject.box import inject_box_transit

def test_grid_parity():
    # 1. Setup synthetic data
    t = np.linspace(0, 10, 5000)
    f = np.ones_like(t)
    true_p = 3.456
    true_t0 = 1.2
    f = inject_box_transit(t, f, true_p, true_t0, 0.1, 0.01)
    f += np.random.normal(0, 0.0001, len(t))
    
    # 2. Shared grid
    periods = np.linspace(2.0, 5.0, 500)
    durations = np.linspace(0.05, 0.15, 3)
    
    # 3. Run CPU
    cpu_res = run_astropy_bls(t, f, periods=periods, durations=durations)
    
    # 4. Run GPU
    gpu_res = run_gpu_bls(t, f, periods, durations)
    
    # 5. Check Parity
    # Note: GPU uses binning, so it won't be EXACTLY identical, but should pick the same best period
    assert abs(cpu_res['period'] - gpu_res['best_period']) < 0.01
    
    # Power correlation (optional check)
    cpu_power = cpu_res['power']
    gpu_power = gpu_res['power'].get()
    
    # The peak should be at the same index or very close
    cpu_best_idx = np.argmax(cpu_power)
    gpu_best_idx = np.argmax(gpu_power)
    assert abs(cpu_best_idx - gpu_best_idx) <= 1
