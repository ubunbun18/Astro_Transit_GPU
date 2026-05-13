# V42 "Astropy Parity" カーネル 仕様書 (CbC形式)

## 1. 目的
Astropyの `bls_fast` (Cython/C実装) とビットレベル（float32/float64精度範囲内）で完全に一致する計算結果を出力するGPUカーネル（V42）を実装する。本カーネルは、速度よりも「完全な科学的パリティ（相関係数 1.000）」を優先したオプションモードとして提供する。

## 2. インターフェース定義
```python
def run_vbls_exact_parity(time_array, flux_matrix, periods, durations, weights_matrix=None, 
                          oversample=10, dtype=cp.float32, target_batch_size=None, period_batch_size=None):
    """
    Astropyと完全に一致するロジックを持つV42カーネルのラッパー。
    
    Args:
        time_array: cp.ndarray (N_DATA)
        flux_matrix: cp.ndarray (N_TARGETS, N_DATA)
        periods: cp.ndarray (N_PERIODS)
        durations: cp.ndarray (N_DURATIONS)
        weights_matrix: cp.ndarray (N_TARGETS, N_DATA) または None
        oversample: int (Astropyデフォルト=10)
        
    Returns:
        dict: {
            "best_period": cp.ndarray (N_TARGETS),
            "best_t0": cp.ndarray (N_TARGETS),
            "best_depth": cp.ndarray (N_TARGETS),
            "best_duration": cp.ndarray (N_TARGETS),
            "snr": cp.ndarray (N_TARGETS) # Astropyの objective="snr" 相当
        }
    """
```

## 3. 詳細ロジック (Astropy C実装の忠実な移植)
Astropy の C 実装では、以下のロジックで各周期ごとのビニングと計算を行っている。これを CUDA カーネルにマッピングする。

**[CPU Astropy のコアアルゴリズム]**
1. 各 `period` に対してループ
2. `n_bins = ceil(oversample * period / min_duration)` を計算
3. `N_DATA` に対してループを回し、`phase = fmod(t * inv_p, 1.0)` を計算して `n_bins` 空間にビニング（`count_bin` と `flux_bin` を蓄積）
4. 各 `duration` に対して `n_dur_bins = round(duration * inv_p * n_bins)` を計算
5. `n_bins` 個の位相をスライドさせながら `sum_in`, `count_in` を計算し、尤度またはSNRを最大化する。

**[V42 GPU CUDA マッピング案]**
- **グリッド構成**: `blockIdx.x` = period_idx, `blockIdx.y` = target_idx
- **スレッド構成**: `threadIdx.x` = N_DATA などを並列処理するワーカー (例: 256 threads)
- **メモリ**: `n_bins` は周期ごとに異なるため、共有メモリを動的確保 `extern __shared__ SCALAR_T smem[]`。最大サイズはハードウェア制限（例: 48KB ~ 10,000 bins）まで。
- **処理フロー**:
  1. 1ブロック = 1周期 × 1ターゲット を担当。
  2. ブロック内のスレッドが協調して `n_bins` を初期化。
  3. ブロック内のスレッドが `N_DATA` を分割して読み込み、Shared Memory の `count_bin`, `flux_bin` に `atomicAdd` でビニング。
  4. Prefix Sum（累積和）をブロック内で協調して計算。
  5. スレッドが開始位相（`start_bin`）を分割して担当し、各 `duration` についてスコア（SNR）を計算。
  6. ブロック内で `max_score` を Reduction し、結果をグローバルメモリに書き出す。

## 4. 自動検証テストケース (Test-Driven)
- `tests/test_v42_parity.py` を作成。
- `oversample=10` を指定し、Astropy(`objective="snr"`) と V42カーネルの `snr` 出力を比較。
- アサーション: `np.corrcoef(cpu_power, gpu_power)[0, 1] > 0.99999`
- ピークSNRの絶対誤差が `1e-5` 以下であることを確認。

## 5. 設計上の注意・制約
- `n_bins` が共有メモリの制限を超える場合（例: 周期が長く min_duration が極端に短い場合）は、Astropy同様に `MAX_BINS`（例: 8000）でキャップをかける。
- `fmod` の挙動など、C言語標準ライブラリと CUDA デバイス関数の浮動小数点演算の極微小な違いにより完全なビット一致にならない可能性があるが、可能な限り同一の演算子を使用する。
