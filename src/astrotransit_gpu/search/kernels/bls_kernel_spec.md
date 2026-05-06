# CbC Specification: Optimized CUDA BLS Kernel

## 1. インターフェース定義 (Python <-> CUDA)

### カーネル名: `bls_kernel`

**入力パラメータ (Kernel Arguments):**
- `d_time` (float*): 時刻配列 (Global Memory)
- `d_flux` (float*): 正規化済み光度配列 (Global Memory)
- `n_data` (int): データ点数
- `d_periods` (float*): 周期グリッド (Global Memory)
- `n_periods` (int): 周期数
- `d_durations` (float*): デュレーショングリッド (Global Memory)
- `n_durations` (int): デュレーション数
- `n_bins` (int): ビン数 (例: 200)
- `t_start` (float): 基準時刻

**出力パラメータ:**
- `d_power` (float*): 各周期の最大スコア
- `d_best_t0` (float*): 各周期の最良エポック
- `d_best_dur` (float*): 各周期の最良デュレーション
- `d_best_depth` (float*): 各周期の最良深さ

## 2. 詳細ロジック（CUDA カーネル内）

```cpp
__global__ void bls_kernel(...) {
    // 1. ブロック割り当て
    // 各ブロックが1つの周期を担当する
    int period_idx = blockIdx.x;
    if (period_idx >= n_periods) return;
    float p = d_periods[period_idx];

    // 2. 共有メモリの準備
    // extern __shared__ float shared_mem[];
    // float* s_counts = &shared_mem[0];
    // float* s_flux_sum = &shared_mem[n_bins];

    // 3. ビンの初期化 (並列)
    for (int b = threadIdx.x; b < n_bins; b += blockDim.x) {
        s_counts[b] = 0;
        s_flux_sum[b] = 0;
    }
    __syncthreads();

    // 4. 位相折り畳みとビン詰め (並列)
    // 各スレッドがデータ点の一部を担当
    for (int i = threadIdx.x; i < n_data; i += blockDim.x) {
        float phase = fmodf(d_time[i] - t_start, p);
        int bin = (int)(phase / p * n_bins);
        if (bin >= n_bins) bin = n_bins - 1;
        atomicAdd(&s_counts[bin], 1.0f);
        atomicAdd(&s_flux_sum[bin], d_flux[i]);
    }
    __syncthreads();

    // 5. BLS スコアの計算 (スライディングウィンドウ)
    // 各スレッドが異なる開始ビンまたは異なるデュレーションを担当
    // ここでは単純化のため、スレッド0が代表して探索（または並列リダクション）
    if (threadIdx.x == 0) {
        float max_pwr = 0;
        // ... (探索ロジック)
        d_power[period_idx] = max_pwr;
        // ...
    }
}
```

## 3. 不変条件 (Invariants)

1. **メモリ安全性**: `bin` インデックスは常に `0` から `n_bins - 1` の範囲に収まる。
2. **同期の正当性**: 共有メモリへの書き込み完了後に必ず `__syncthreads()` を呼び出す。
3. **リソース制約**: `n_bins` と共有メモリのサイズが GPU の制限を超えない。

## 4. テストケース

### 4.1 精度テスト
- Stage A1 (CuPy) の結果と一致すること。

### 4.2 性能テスト
- 1000周期の探索が 1秒以内（初期化を除く）で完了すること。
- データの点数が増えても、ブロック並列化によりスケールすること。
