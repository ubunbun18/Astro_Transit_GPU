# Comprehensive Performance Benchmark Specification

## 1. 計測の目的
AstroTransit-GPU V39 カーネルの計算スループットを多角的にプロファイリングし、Blackwell アーキテクチャ上での限界性能と、実運用における最適パラメータを特定する。

## 2. 計測次元 (Matrix of Benchmarks)

### 2.1 周期グリッド・スケーラビリティ
周期グリッド数（$N_{periods}$）を変化させた際のスループットを計測する。
- 10,000 (Standard)
- 100,000 (High-precision)
- 1,000,000 (Extreme-search)

### 2.2 バッチサイズ最適化 (Blackwell Hyper-batching)
`target_batch_size` と `period_batch_size` の組み合わせによるスループットの変化。
- Target Batches: 1, 128, 1024, 4096, 8192
- Period Batches: 1000, 10000, 25000, 50000

### 2.3 データ密度 (N_DATA)
ライトカーブのデータ点数による影響。
- 1,312 (TESS QLP / 2-min)
- 10,000 (TESS 20-sec cadence 相当)

---

## 3. 評価指標 (Metrics)

| 指標 | 単位 | 内容 |
| :--- | :--- | :--- |
| **LC/sec** | 天体/秒 | 1秒間に処理されるライトカーブの総数 |
| **GCPS** | Giga-checks/s | 1秒間に評価される試行回数 ($N_{targets} \times N_{periods} \times N_{durations} / Time$) |
| **Occupancy** | % | GPU 演算器の稼働率 (SM 飽和度) |
| **I/O Bound Ratio** | % | 全実行時間に対するデータ転送の割合 |

---

## 4. 検証プロセス
1. **Warm-up**: GPU クロックを安定させるためのダミー実行 (1000 periods)。
2. **Cold Run**: 初回コンパイル時間を含む計測。
3. **Hot Run**: キャッシュされたカーネルによる純粋な計算時間の 3回平均。
4. **Stress Test**: VRAM 限界付近（200,000天体バッチ等）での安定性確認。

## 5. 期待されるアウトプット
- 最も効率的なバッチサイズの組み合わせの特定。
- 周期グリッド数に対する計算時間の線形性の確認。
- RTX 5060 Ti における理論ピーク性能に対する到達率の算出。
