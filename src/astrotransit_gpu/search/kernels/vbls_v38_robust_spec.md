# CbC Specification: V38 "Apex Predator (Robust)" CUDA Kernel

## 1. インターフェース定義

### カーネル名: `vbls_v38_robust_kernel`

**入力パラメータ:**
- `flux_matrix` (float*): 各天体の光度データ ($N_{target} \times N_{data}$)
- `dt_array` (float*): 時刻データ ($N_{data}$)
- `period_pairs` (float2*): 周期グリッド ($N_{periods}$)
- `global_rms` (float*): 各天体の標準偏差 ($\sigma$) ($N_{target}$)
- `n_periods` (int): 周期数
- `t_start` (float): 基準時刻

**出力パラメータ:**
- `global_max_power` (float*): 最大 SNR (正規化済み)
- `global_best_t0`, `global_best_dur`, `global_best_depth`, `global_best_period`

---

## 2. 詳細ロジック（擬似コード）

### 2.1 外れ値除去 (Outlier Clipping)
各スレッドがデータを読み込む際、当該天体の `rms` を使用して値を制限する。
```cpp
float f = flux_matrix[offset + i];
float rms = global_rms[target_idx];
// 10-sigma クリッピング
float limit = 10.0f * rms;
if (f > 1.0f + limit) f = 1.0f + limit;
if (f < 1.0f - limit) f = 1.0f - limit;
```

### 2.2 SNR 正規化とオーバーフロー防止
従来の Cross-multiplication スコア `(numer^2 / denom)` ではなく、直接 SNR を計算する。
```cpp
// numer = N_in * N_out * (mean_in - mean_out)
// この numer は float の範囲内に収まる (clipped f ならば)
float numer = cur_f * total_c - cur_c * total_f;

// SNR = numer / (rms * sqrt(cur_c * out_c * total_c))
float noise_term = rms * sqrtf(cur_c * (total_c - cur_c) * total_c);
float snr = (noise_term > 1e-10f) ? (numer / noise_term) : 0.0f;
```
※ `total_c` は `N_DATA` (定数)。

### 2.3 不変条件 (Invariants)
1. **SNRの非負性**: トランジット（減光）のみを対象とする場合、`mean_in < mean_out` (すなわち `numer < 0`) のケースを重視するが、一般には `abs(snr)` を使用。
2. **ゼロ除算の防止**: `noise_term` が極めて小さい場合は SNR を 0 とする。

---

## 3. テストケース

### 3.1 正常系
- 清浄なサイン波データにおいて、既知の SNR と一致すること。
- `rms` が正しく反映され、異なるノイズレベルの天体間で SNR が比較可能であること。

### 3.2 堅牢性テスト
- データに $10^{30}$ のような巨大な外れ値が含まれていても、`inf` を出力せず、適切にクリップされること。

### 3.3 境界値テスト
- `rms` が 0 または負数の場合（事前チェックで弾くべきだが、カーネル内でも保護する）。
