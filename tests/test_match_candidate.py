import pytest
from astrotransit_gpu.validate.match import match_candidate

def test_direct_match():
    # Perfect match
    res = match_candidate(5.0, 100.0, 5.0, 100.0)
    assert res['is_match'] is True
    assert res['match_type'] == 'direct'

def test_t0_shift_match():
    # T0 shifted by exactly one period
    res = match_candidate(5.0, 105.0, 5.0, 100.0)
    assert res['is_match'] is True
    
    # T0 shifted by small amount within tolerance
    res = match_candidate(5.0, 100.05, 5.0, 100.0, t0_tol=0.1)
    assert res['is_match'] is True

def test_harmonic_match():
    # Detected 2 * P_true
    res = match_candidate(10.0, 100.0, 5.0, 100.0)
    assert res['is_match'] is True
    assert res['match_type'] == 'harmonic'

def test_subharmonic_match():
    # Detected P_true / 2
    res = match_candidate(2.5, 100.0, 5.0, 100.0)
    assert res['is_match'] is True
    assert res['match_type'] == 'subharmonic'

def test_mismatch_period():
    res = match_candidate(7.0, 100.0, 5.0, 100.0)
    assert res['is_match'] is False

def test_harmonic_match_shifted_phase():
    # Detected 2 * P_true. 
    # Even if t0_detected is shifted by P_true, it should still match if it's on the series.
    p_true = 5.0
    t0_true = 100.0
    p_det = 10.0
    t0_det = 105.0 # This is t0_true + p_true
    
    res = match_candidate(p_det, t0_det, p_true, t0_true)
    assert res['is_match'] is True
    assert res['match_type'] == 'harmonic'

def test_optional_t0():
    # When require_t0 is False, match should be True even if T0 is wrong
    res = match_candidate(5.0, 102.5, 5.0, 100.0, require_t0=False)
    assert res['is_match'] is True
