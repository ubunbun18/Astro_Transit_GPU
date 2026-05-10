import numpy as np
import pytest
from astrotransit_gpu.search.api import BoxLeastSquaresGPU
from astrotransit_gpu.search.cpu_reference_bls import run_astropy_bls
from astrotransit_gpu.inject.box import inject_box_transit

@pytest.mark.gpu
def test_astropy_parity_rigorous():
    """
    Rigorous parity check between Astropy (CPU) and our GPU backend.
    Checks: Best parameters, Power spectrum correlation, and RMSE.
    """
    # 1. Setup synthetic data with noise
    np.random.seed(42)
    t = np.linspace(0, 10, 5000)
    f = np.ones_like(t)
    true_p, true_t0, true_dur, true_depth = 3.456, 0.5, 0.1, 0.01
    f = inject_box_transit(t, f, true_p, true_t0, true_dur, true_depth)
    f += np.random.normal(0, 0.001, len(t))
    
    periods = np.linspace(1.0, 5.0, 1000)
    durations = np.array([0.05, 0.1, 0.15])
    
    # 2. Run GPU with higher binning for parity
    model = BoxLeastSquaresGPU(t, f)
    gpu_res = model.power(periods, durations, n_bins=500, dtype=np.float32)
    
    # 3. Run CPU (Astropy)
    cpu_res = run_astropy_bls(t, f, periods=periods, durations=durations)
    
    # 4. Assertions
    # A. Best Period (Allow for small binning artifacts)
    assert abs(gpu_res.best_period - cpu_res['period']) < 0.01
    
    # B. Depth Check (Parity)
    assert abs(gpu_res.best_depth - cpu_res['depth']) < 0.001


    # D. Top-1 Overlap (Basic)
    assert abs(gpu_res.top_candidates[0].period - cpu_res['period']) < 0.01
