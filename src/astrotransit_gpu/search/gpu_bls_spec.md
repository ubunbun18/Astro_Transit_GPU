# CbC Specification: GPU BLS Search Core

## 1. インターフェース定義

### 関数名: `run_gpu_bls`

**入力:**
- `time` (cupy.ndarray): 時刻配列 (1D, float64/float32)
- `flux` (cupy.ndarray): 光度配列 (1D, float64/float32, 正規化済み)
- `periods` (cupy.ndarray): 探索対象の周期グリッド (1D)
- `durations` (cupy.ndarray): 探索対象のデュレーショングリッド (1D)

**出力:**
- `dict`: 以下のキーを持つ辞書
    - `best_period`: 最良の周期
    - `best_t0`: 最良のエポック
    - `best_duration`: 最良のデュレーション
    - `best_depth`: 最良の深さ
    - `power`: 全周期のパカースペクトル (cupy.ndarray)
    - `snr`: 最良の結果のSNR

## 2. 詳細ロジック（擬似コード）

```python
def run_gpu_bls(time, flux, periods, durations):
    # 1. 前提条件チェック
    assert len(time) == len(flux)
    assert len(periods) > 0
    assert len(durations) > 0

    # 2. 結果格納用バッファの初期化 (GPU上)
    # n_periods = len(periods)
    # power_array = cp.zeros(n_periods)

    # 3. 周期グリッド全体にわたるループ
    # 実際の実装では CuPy のブロードキャストや RawKernel を検討するが、
    # Stage A1 ではまずは正しさを重視した CuPy 実装とする。
    
    for i, p in enumerate(periods):
        # 簡易的な BLS スコア計算（各周期・デュレーションに対して）
        # 1. 位相空間へのマッピング
        # 2. 指定されたデュレーション窓での畳み込み
        # 3. 最大値を記録
        ...

    return results
```

## 3. 不変条件 (Invariants)

1. **データ整合性**: `time` と `flux` の形状は常に一致する必要がある。
2. **計算範囲**: 検出される `best_period` は入力された `periods` 配列の中に存在するか、その範囲内にあること。
3. **正値性**: `power` は正の実数であること。

## 4. テストケース

### 4.1 境界値テスト
- 配列長が最小限の場合。
- `periods` または `durations` が1つの場合。

### 4.2 契約違反テスト
- 入力配列の長さ不一致。
- 入力に NaN が含まれる場合。

### 4.3 比較テスト
- Astropy BLS との比較（数値誤差が 1e-5 以内であること）。
