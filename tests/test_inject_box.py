import numpy as np
import pytest
from astrotransit_gpu.inject.box import inject_box_transit

def test_inject_box_depth():
    t = np.linspace(0, 10, 1000)
    f = np.ones_like(t)
    p = 5.0
    t0 = 2.5
    dur = 0.2
    depth = 0.01
    
    f_injected = inject_box_transit(t, f, p, t0, dur, depth)
    
    # Check if some points are actually dropped
    assert np.any(f_injected < 1.0)
    assert np.min(f_injected) == pytest.approx(1.0 - depth)

def test_inject_box_timing():
    t = np.array([2.5]) # Exactly at t0
    f = np.array([1.0])
    p = 5.0
    t0 = 2.5
    dur = 0.2
    depth = 0.1
    
    f_injected = inject_box_transit(t, f, p, t0, dur, depth)
    assert f_injected[0] == pytest.approx(0.9)
    
    # Half period away
    t2 = np.array([5.0])
    f2 = np.array([1.0])
    f_injected2 = inject_box_transit(t2, f2, p, t0, dur, depth)
    assert f_injected2[0] == 1.0
