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

def test_mismatch_phase():
    # Period matches, but T0 is shifted by P/2
    res = match_candidate(5.0, 102.5, 5.0, 100.0, t0_tol=0.1)
    assert res['is_match'] is False
