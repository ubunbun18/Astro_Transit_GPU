import pytest
import pandas as pd
import numpy as np
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.large_scale_bench import LargeScaleValidator

@pytest.fixture
def sample_results():
    return pd.DataFrame({
        'tic_id': ['1001', '1002', '1003', '1004'],
        'period': [10.0, 20.0, 5.0, 30.0],
        't0': [100.0, 105.0, 110.0, 120.0],
        'power': [20.0, 25.0, 5.0, 18.0],
        'depth': [0.01, 0.02, 0.005, 0.015]
    })

@pytest.fixture
def sample_toi():
    return pd.DataFrame({
        'tid': ['1001', '1002', '1005'],
        'pl_orbper': [10.01, 10.0, 15.0], # 1001: direct, 1002: harmonic
        'pl_tranmid': [100.0, 105.0, 110.0],
        'st_tmag': [10.0, 12.0, 15.0],
        'st_teff': [5800, 4500, 3500], # G, K, M type
        'toi': ['TOI 1.01', 'TOI 2.01', 'TOI 5.01']
    })

def test_validation_normal_case(sample_results, sample_toi):
    """正常系: direct match と harmonic match が正しく識別されるか"""
    validator = LargeScaleValidator(p_tol=0.03, power_threshold=10.0)
    summary, matches, new_cands = validator.validate_results(sample_results, sample_toi)
    
    assert summary['toi_in_sample'] == 2  # 1001, 1002
    assert summary['recovered_toi'] == 2
    assert summary['recovery_rate'] == 1.0
    
    # 1001 は direct match
    assert matches[matches['tic_id'] == '1001']['match_type'].iloc[0] == 'direct'
    # 1002 は 検出20.0 vs カタログ10.0 なので harmonic
    assert matches[matches['tic_id'] == '1002']['match_type'].iloc[0] == 'harmonic'

def test_new_candidates_extraction(sample_results, sample_toi):
    """正常系: カタログ未登録の高Power天体が抽出されるか"""
    validator = LargeScaleValidator(p_tol=0.01, power_threshold=10.0)
    _, _, new_cands = validator.validate_results(sample_results, sample_toi)
    
    # 1004 は Power 18.0 でカタログにない
    assert '1004' in new_cands['tic_id'].values
    # 1003 は Power 5.0 なので閾値以下
    assert '1003' not in new_cands['tic_id'].values

def test_boundary_power_threshold(sample_results, sample_toi):
    """境界値テスト: Powerがちょうど閾値の場合"""
    validator = LargeScaleValidator(power_threshold=18.0)
    _, _, new_cands = validator.validate_results(sample_results, sample_toi)
    
    # 18.0 は > 18.0 でないので含まれない (仕様上の > vs >=)
    # 実装は > なので 18.0 は含まれないはず
    assert '1004' not in new_cands['tic_id'].values

def test_invariant_recovery_rate_range(sample_results, sample_toi):
    """不変条件検証: 回収率が 0.0 ~ 1.0 の範囲に収まるか"""
    validator = LargeScaleValidator()
    summary, _, _ = validator.validate_results(sample_results, sample_toi)
    assert 0.0 <= summary['recovery_rate'] <= 1.0

def test_empty_results():
    """異常系: 結果が空の場合"""
    validator = LargeScaleValidator()
    summary, matches, new_cands = validator.validate_results(pd.DataFrame(), pd.DataFrame())
    assert summary['total_targets'] == 0
    assert matches.empty
    assert new_cands.empty

def test_contract_violation_negative_period():
    """契約違反テスト: 周期が負数の場合（match_candidate側で処理）"""
    res = pd.DataFrame({'tic_id': ['999'], 'period': [-1.0], 't0': [0], 'power': [20]})
    toi = pd.DataFrame({'tid': ['999'], 'pl_orbper': [10.0], 'pl_tranmid': [0], 'st_tmag': [12.0], 'st_teff': [5000]})
    
    validator = LargeScaleValidator()
    _, matches, _ = validator.validate_results(res, toi)
    assert not matches['is_match'].iloc[0]
