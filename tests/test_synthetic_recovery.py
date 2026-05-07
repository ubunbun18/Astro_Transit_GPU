import numpy as np
import pytest
from astrotransit_gpu.search.gpu_bls import run_gpu_bls
from astrotransit_gpu.inject.box import inject_box_transit
from astrotransit_gpu.validate.match import match_candidate

@pytest.mark.gpu
def test_full_synthetic_recovery():
    # 1. Generate data
    t = np.linspace(0, 20, 5000)
    f = np.ones_like(t)
    true_p = 7.432
    true_t0 = 1.5
    true_dur = 0.15
    true_depth = 0.01
    
    f = inject_box_transit(t, f, true_p, true_t0, true_dur, true_depth)
    np.random.seed(42)
    f += np.random.normal(0, 0.001, len(t))
    
    # 2. Search
    periods = np.linspace(5.0, 10.0, 1000)
    durations = np.linspace(0.05, 0.25, 5)
    
    res = run_gpu_bls(t, f, periods, durations)
    
    # 3. Match
    match = match_candidate(res['best_period'], res['best_t0'], true_p, true_t0)
    
    assert match['is_match'] is True
    assert match['match_type'] == 'direct'
    assert abs(res['best_period'] - true_p) < 0.01
