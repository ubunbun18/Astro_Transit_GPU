# CbC Specification: V39 "Apex Predator (Weight-aware)" CUDA Kernel

## 1. インターフェース定義

### カーネル名: `vbls_v39_weighted_kernel`

**入力パラメータ:**
- `flux_matrix` (float*): 各天体の光度データ
- `weights_matrix` (float*): 各データ点の重み ($w = 1/\sigma^2$)。パディング領域は $w=0$。
- `dt_array` (float*): 時刻データ
- `period_pairs` (float2*): 周期グリッド
- `n_periods` (int): 周期数
- `t_start` (float): 基準時刻

**出力パラメータ:**
- `global_max_power` (正規化 SNR)
- `global_best_t0`, `global_best_dur`, `global_best_depth`, `global_best_period`

---

## 2. 詳細ロジック（CUDA カーネル内）

### 2.1 重み付きビニング (Weighted Binning)
各ビンにおいて、データのカウント数 $N$ の代わりに「重みの合計 $W$」と「重み付き光度の合計 $WF$」を計算する。
```cpp
float w = weights_matrix[offset + i];
float f = flux_matrix[offset + i];

// パディング (w=0) は加算されても結果に影響しない
atomicAdd(&s_w[warp_id][b + 1], w);
atomicAdd(&s_wf[warp_id][b + 1], w * f);
```

### 2.2 重み付き SNR 計算 (Weighted SNR)
全データの重み合計を $total\_W$、重み付き光度合計を $total\_WF$ とする。
特定のウィンドウ内の合計を $S\_W$, $S\_WF$ とすると：

$$SNR = \frac{|total\_W \cdot S\_WF - S\_W \cdot total\_WF|}{\sqrt{S\_W \cdot (total\_W - S\_W) \cdot total\_W}}$$

この式は、各点のノイズが $\sigma_i = 1/\sqrt{w_i}$ である場合の統計的に最適な SNR となる。

### 2.3 外れ値の保護
V38 同様、極端な $f$ をクリップする。ただし、クリッピングの閾値は重み $w$ から算出される局所的な $\sigma = 1/\sqrt{w}$ を用いる。

---

## 3. 不変条件 (Invariants)

1. **ゼロ除算の回避**: $S\_W$ または $(total\_W - S\_W)$ が 0 に近い場合、SNR を 0 とする。
2. **重みの非負性**: $w$ は常に 0 以上である。
3. **統計的整合性**: すべての $w=1.0$ の場合、結果は従来のカウントベースの SNR と一致する。

---

## 4. 期待される成果
- **パディングの影響除去**: 観測の隙間が SNR 計算に悪影響を与えなくなる。
- **既知惑星の回収**: TOI 101 等の信号が、ノイズに埋もれず SNR 10 以上の有意なピークとして検出される。
