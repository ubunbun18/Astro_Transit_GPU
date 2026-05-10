import pytest
import pandas as pd
import numpy as np
import os
from astrotransit_gpu.vet.harmonics import group_harmonics
from astrotransit_gpu.vet.ranking import calculate_vetting_scores

def test_harmonic_grouping():
    # Setup dummy data with harmonics
    data = {
        'tic_id': [1, 1, 1, 2, 2],
        'period': [5.0, 10.001, 2.501, 3.0, 9.0], # 1: 5.0, 2P=10.0, P/2=2.5. 2: 3.0, 3P=9.0
    }
    df = pd.DataFrame(data)
    
    res = group_harmonics(df, tolerance=0.01)
    
    # Check TIC 1
    tic1 = res[res['tic_id'] == 1]
    assert len(tic1['harmonic_group_id'].unique()) == 1
    # Sorted canonical is 2.501. 5.0 is 2:1, 10.0 is 4:1.
    assert set(tic1['harmonic_relation']) == {"1:1", "2:1", "4:1"}
    
    # Check TIC 2
    tic2 = res[res['tic_id'] == 2]
    assert len(tic2['harmonic_group_id'].unique()) == 1
    assert set(tic2['harmonic_relation']) == {"1:1", "3:1"}

def test_vetting_scores():
    data = {
        'tic_id': [1, 2, 3],
        'power': [1e9, 1e8, 1e9],
        'period': [5.0, 5.0, 5.0],
        'duration': [0.1, 0.1, 3.0], # 3 is physically impossible (duration > period/2)
        'known_type': ['unknown', 'unknown', 'unknown']
    }
    df = pd.DataFrame(data)
    
    config = {
        'snr_weight': 1.0,
        'snr_norm': 1e9,
        'plausibility_weight': 1.0
    }
    
    res = calculate_vetting_scores(df, config=config)
    
    # TIC 1: Max SNR, Good Plausibility -> ~1.0
    assert res.iloc[0]['vetting_score'] > 0.9
    
    # TIC 2: Low SNR -> lower score
    assert res.iloc[1]['vetting_score'] < res.iloc[0]['vetting_score']
    
    # TIC 3: High SNR but Impossible Plausibility -> very low score
    assert res.iloc[2]['vetting_score'] < 0.2

def test_catalog_bonus_penalty():
    data = {
        'tic_id': [1, 2, 3],
        'power': [0.5e9, 0.5e9, 0.5e9],
        'period': [5.0, 5.0, 5.0],
        'duration': [0.1, 0.1, 0.1],
        'known_type': ['TOI', 'EB', 'unknown']
    }
    df = pd.DataFrame(data)
    
    config = {
        'snr_weight': 1.0,
        'snr_norm': 1e9,
        'known_toi_bonus': 0.2,
        'known_eb_penalty': -0.8
    }
    
    res = calculate_vetting_scores(df, config=config)
    
    # TOI should be highest, EB should be lowest
    assert res.iloc[0]['vetting_score'] > res.iloc[2]['vetting_score']
    assert res.iloc[1]['vetting_score'] < 0.3
