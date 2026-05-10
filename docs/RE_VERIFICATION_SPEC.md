# Re-Verification Specification: AstroTransit-GPU Accuracy & Speed

## 1. 検証の目的
V39 "Apex Predator (Weight-aware)" カーネルの科学的正確性（Astropyとの数値一致性）および計算パフォーマンス（スループット）を、厳密な条件下で再確認する。

## 2. 科学的正確性の検証 (Scientific Accuracy)

### 2.1 基準値 (Ground Truth)
`astropy.timeseries.BoxLeastSquares` の結果を真値とする。
※ `depth_snr` および `power` (log-likelihood ratio) の両面で比較を行う。

### 2.2 許容誤差範囲 (Tolerance)
| 項目 | 許容誤差 | 備考 |
| :--- | :--- | :--- |
| Period | < 10^-6 days | 周期グリッド密度の影響を除く |
| T0 | < 10^-3 days | ビンサイズ (N_BINS=128) の分解能程度 |
| SNR | < 1.5% (相対誤差) | 浮動小数点演算とビニングの丸め誤差 |
| Depth | < 5.0% (相対誤差) | ビニングによる深さの平均化の影響 |

### 2.3 検証シナリオ
- **Scenario A (Ideal)**: ノイズなし、欠損なしの完全な矩形トランジット。
- **Scenario B (Noisy)**: ガウスノイズ重畳、SNR=10 程度のトランジット。
- **Scenario C (Sparse/Gaps)**: 50% のデータが欠損（Weight=0）している状態。
- **Scenario D (Real Data)**: 既知の TOI (TIC 91450443 等) を用いた実測比較。

---

## 3. 速度の検証 (Computational Performance)

### 3.1 計測指標
- **Throughput (LC/sec)**: 1秒間に処理できるライトカーブの数。
- **Giga-Checks/sec (GCPS)**: 1秒間に評価される $\{Period \times T0 \times Duration\}$ の試行回数。

### 3.2 目標値 (Blackwell RTX 5060 Ti 以上を想定)
- **Target**: > 300 LC/sec (100,000周期/LC の場合)
- **Computation**: > 50 GCPS

---

## 4. 検証ロジック (Pseudo-code)

### 4.1 数値的一致性テスト
```python
def test_scientific_parity():
    lc = generate_synthetic_lc(period=3.5, duration=0.1, depth=0.01, noise=0.001)
    # 1. Run Astropy
    res_astropy = run_astropy_bls(lc)
    # 2. Run GPU V39
    res_gpu = run_gpu_v39(lc)
    # 3. Assert matches within tolerance
    assert abs(res_astropy.period - res_gpu.period) < 1e-6
    assert relative_error(res_astropy.snr, res_gpu.snr) < 0.01
```

### 4.2 ストレス計測 (スループット)
```python
def test_throughput():
    batch_size = 1000
    start = time.time()
    run_gpu_v39_batch(large_dataset, batch_size)
    elapsed = time.time() - start
    print(f"Throughput: {batch_size / elapsed} LC/sec")
```

---

## 5. 自動検証の実施
- `pytest` 形式のテストコードを生成し、境界値、パディング、極端な SNR 条件を網羅する。
- 検証結果を `reports/THOROUGH_VERIFICATION_REPORT.md` に出力する。
