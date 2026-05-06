# CbC Specification: Planet Matching Logic

## 1. インターフェース定義

### 関数名: `match_candidate`

**入力:**
- `p_detected` (float): 検出された周期
- `t0_detected` (float): 検出されたエポック
- `p_true` (float): カタログの真の周期
- `t0_true` (float): カタログの真のエポック
- `p_tol` (float): 周期の相対許容誤差 (デフォルト 0.01)
- `t0_tol` (float): エポックの絶対許容誤差 (デフォルト 0.1日)

**出力:**
- `dict`:
    - `is_match` (bool): 一致するかどうか
    - `match_type` (str): 'direct', 'harmonic', 'subharmonic', 'none'
    - `p_diff` (float): 周期の差

## 2. 詳細ロジック（擬似コード）

```python
def match_candidate(p_detected, t0_detected, p_true, t0_true):
    # 1. 直接一致の確認
    if abs(p_detected - p_true) / p_true < p_tol:
        return {'is_match': True, 'match_type': 'direct'}
    
    # 2. 高調波 (Harmonics) の確認
    # 例: P_det = 2 * P_true, 3 * P_true...
    for n in [2, 3]:
        if abs(p_detected - n * p_true) / (n * p_true) < p_tol:
            return {'is_match': True, 'match_type': 'harmonic'}
    
    # 3. 低調波 (Sub-harmonics) の確認
    # 例: P_det = P_true / 2, P_true / 3...
    for n in [2, 3]:
        if abs(p_detected - p_true / n) / (p_true / n) < p_tol:
            return {'is_match': True, 'match_type': 'subharmonic'}
            
    return {'is_match': False, 'match_type': 'none'}
```

## 3. 不変条件 (Invariants)

1. **対称性**: `p_tol` が十分に小さい場合、直接一致の判定は対称的であるべき。
2. **非負性**: 周期は常に正数であること。

## 4. テストケース

### 4.1 境界値テスト
- `p_detected` が `p_true` と完全に一致する場合。
- 誤差がちょうど `p_tol` の場合。

### 4.2 契約違反テスト
- 周期が負数または 0 の場合。

### 4.3 正常系
- `p_det = 10.0`, `p_true = 10.001` -> direct match
- `p_det = 20.0`, `p_true = 10.0` -> harmonic match
- `p_det = 5.0`, `p_true = 10.0` -> subharmonic match
