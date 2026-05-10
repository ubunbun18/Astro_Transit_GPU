# AstroTransit-GPU 設計書  
## TESS/Kepler公開光度曲線から、既知惑星・人工注入惑星を高速に再検出する、再現可能なGPUトランジット探索基盤

**作成日**: 2026-05-10 (v1.3.0)
**対象**: 個人研究・OSS・学術発表・ASCL登録・将来的な論文化  
**採用する新規性**:  
- **新規性A**: GPU高速化そのもの  
- **新規性B**: 再現可能ベンチマーク基盤  
- **新規性C**: GPU探索 + injection/recovery の自動評価  

---

## 0. ひとことで言うと

**AstroTransit-GPU** は、TESS/Kepler の公開光度曲線を使って、

1. 既知惑星を再検出し、
2. 人工注入した惑星信号を回収し、
3. CPU版とGPU版の速度・精度・検出率を公平に比較し、
4. その結果をJSON/HTML/Markdownで再現可能に出力する、

**GPU高速トランジット探索・検証基盤**である。

単なる「GPU版BLS」ではなく、

> **データ取得 → 前処理 → GPU探索 → 既知カタログ照合 → injection/recovery → ベンチマーク → レポート生成**

までを一気通貫で行うことを目標とする。

---

## 1. 背景

### 1.1 なぜこのテーマか

TESS/Kepler の光度曲線データは一般公開されており、Lightkurve を使えば比較的簡単に取得・解析できる。  
また、NASA Exoplanet Archive には既知惑星、TOI、KOI、TCE などの照合用カタログが整備されている。  
さらに Kepler には transit injection 関連データがあり、探索器の回収率評価に使える。

このため、他の天文学高速化テーマと比べて以下の利点がある。

| 条件 | TESS/Keplerトランジット探索 |
|---|---|
| 一般人でもデータ取得しやすい | 強い |
| 正解確認しやすい | 強い |
| 既存CPU基準がある | 強い |
| GPU高速化の効果が測れる | 強い |
| 学術的評価指標が作りやすい | 強い |
| デモとして説明しやすい | 強い |

### 1.2 既存でできていること

既存の代表的な部品はすでに強い。

| 役割 | 既存のもの | できていること |
|---|---|---|
| データ取得 | Lightkurve / MAST | Kepler/TESS light curve や target pixel file の取得 |
| 標準データ製品 | TESS SPOC / TESS-SPOC / QLP / eleanor | light curve 作成済み製品 |
| CPU BLS | Astropy BoxLeastSquares / Lightkurve | BLS探索、period/duration/epoch推定 |
| 大量時系列解析 | VARTOOLS | CLIによるbatch/parallel処理 |
| 高感度探索 | TLS | 物理的な transit-like model による探索 |
| GPU探索 | cuvarbase / CETRA | GPU BLSやGPU向け高感度探索 |
| GPU transit model | PyTransit | transit modelのCPU/GPU高速評価 |
| 既知惑星照合 | NASA Exoplanet Archive | confirmed planets, TOI, KOI, TCE |
| injection評価 | Kepler injection products | ground truthに近い検出率評価材料 |

### 1.3 既存で弱いこと

既存ツールは部品としては強いが、以下はまだ弱い。

1. **TESS/Kepler用の現代的GPU batch探索基盤**
2. **CPU/GPUの数値一致・精度差分を標準で出す仕組み**
3. **公式カタログ照合まで含む一気通貫ワークフロー**
4. **injection/recoveryを標準で回す仕組み**
5. **前処理ごとの検出率比較**
6. **Docker/CI/config固定の再現可能ベンチマーク**
7. **HTML/JSON/Markdownの自動科学レポート**
8. **Lightkurve並みに使いやすいGPU API**
9. **Astropy BLS互換のGPU backend**
10. **multi-GPU / cloud GPU 対応**

AstroTransit-GPU は、この隙間を狙う。

---

## 2. 研究・開発コンセプト

### 2.1 中核コンセプト

> **Lightkurveの使いやすさ + GPU探索の速度 + NASA Archive/Kepler injectionによる自動検証**

を1つの再現可能な基盤としてまとめる。

### 2.2 作るべきもの

作るべきものは、単独のアルゴリズムではなく **検証可能な高速探索基盤** である。

つまり主張はこうなる。

> AstroTransit-GPU は、TESS/Kepler光度曲線に対し、GPU上で大規模トランジット探索を行い、Astropy/Lightkurve等のCPU基準と数値比較し、既知惑星・TOI・KOI・Kepler injectionを使って回収率を自動評価する、再現可能なベンチマーク基盤である。

---

## 3. 採用する新規性

# 新規性A: GPU高速化そのもの

## A.1 狙い

BLS系またはBLS互換探索をGPUで高速化する。

高速化対象は以下。

```text
for star in stars:
  for period in periods:
    for duration in durations:
      for phase/bin in folded_light_curve:
        compute score
```

これをGPUで、

```text
parallel over:
  star × period × duration × phase/bin
```

として処理する。

## A.2 なぜBLSから始めるか

BLSは既存CPU基準が強く、Astropy/Lightkurveで動作確認しやすい。  
また、探索結果を既知惑星の period / epoch / duration と比較しやすい。

最初にBLS互換実装を作ることで、以下が可能になる。

- CPU版との一致確認
- 数値誤差の評価
- GPU高速化の速度評価
- 既知惑星再検出ベンチ
- injection/recoveryベンチ

CETRAのような高感度アルゴリズムを最初から再実装するより、BLS互換GPU backendの方が正しさ確認がしやすい。

## A.3 GPU実装の段階

### A.3.1 Stage A0: CPU baseline

- Astropy BoxLeastSquares
- Lightkurve BLS
- 自作NumPy/Numba BLS簡易版

目的：

- データ処理の正しさ確認
- period grid / duration grid / objectiveの仕様固定
- GPU版のreference作成

### A.3.2 Stage A1: Naive GPU BLS

最初はシンプルでよい。

- CuPyまたはPyTorchで試作
- 1 light curveごとにGPUに転送
- period × duration を並列評価
- 速度より正しさ重視

### A.3.3 Stage A2: Batched GPU BLS

複数 light curve をまとめて処理する。

- shape: `[n_stars, n_time]`
- period grid 共通
- duration grid 共通
- mask / quality flag 対応
- NaN処理
- gap処理

### A.3.4 Stage A3: CUDA C++ kernel

本命。

- pybind11 または CuPy RawKernel
- shared memory
- coalesced memory access
- stream並列
- period grid chunking
- duration grid chunking
- phase binning最適化
- result top-k抽出

### A.3.5 Stage A4: Fused preprocessing + BLS

GPU上で一部前処理も融合する。

候補：

- mean/median normalization
- sigma clipping
- simple detrending
- quality mask適用
- transit score計算
- top-k candidate extraction

メモリ読み書きを減らすことが狙い。

## A.4 Aの成果物

- `astrotransit_gpu.gpu_bls`
- `astrotransit_gpu.cpu_bls`
- `astrotransit_gpu.compare`
- CPU/GPU一致テスト
- speed benchmark
- GPU kernel profiling report

---

# 新規性B: 再現可能ベンチマーク基盤

## B.1 狙い

単なる高速化ではなく、誰でも同じ条件で再実行できるベンチマーク基盤を作る。

研究としては、ここが非常に重要。

## B.2 ベンチマークで固定するもの

以下をすべてconfigに保存する。

```yaml
run_id: 2026-05-07-tess-known-001
mission: tess
data_source: lightkurve
flux_column: pdcsap_flux
targets:
  source: nasa_exoplanet_archive_toi
  n_targets: 1000
preprocessing:
  remove_nans: true
  quality_mask: default
  normalize: median
  flatten:
    method: lightkurve
    window_length: 401
search:
  method: gpu_bls
  period_min_days: 0.5
  period_max_days: 50.0
  duration_grid_hours: [1, 2, 4, 8]
  objective: snr
  oversample: 10
validation:
  catalog: toi
  period_tolerance: 0.01
  harmonic_match: true
hardware:
  gpu: auto
  cpu_threads: auto
random_seed: 42
```

## B.3 ベンチマーク対象

### B.3.1 Known Planet Benchmark

既知惑星を再検出する。

入力：

- NASA Exoplanet Archive `ps` / `pscomppars`
- TESS TOI table
- Kepler KOI table

評価：

- period一致率
- epoch一致率
- duration一致率
- depth一致率
- harmonic match率
- missed target一覧
- false positive一覧

### B.3.2 Injection Benchmark

人工注入信号を回収する。

入力：

- 自前injection
- Kepler injection products

評価：

- recovery rate
- completeness map
- false alarm rate
- period error
- depth error
- duration error
- detection SNR
- runtime

### B.3.3 CPU/GPU Numerical Benchmark

CPUとGPUの数値差分を見る。

比較：

- Astropy BLS fast
- Astropy BLS slow small case
- 自作CPU
- 自作GPU
- optional: cuvarbase
- optional: CETRA

評価：

- best period一致
- top-k period overlap
- power array差分
- depth差分
- transit_time差分
- float32 / float64差分

### B.3.4 Preprocessing Benchmark

前処理が検出率に与える影響を見る。

比較：

- SAP
- PDCSAP
- Lightkurve flatten
- CBV correction
- QLP light curve
- TESS-SPOC light curve
- eleanor corrected light curve

評価：

- recovery rate
- false positive rate
- long-period transit retention
- shallow transit sensitivity
- runtime

## B.4 出力形式

### B.4.1 JSON

機械可読。

```json
{
  "run_id": "2026-05-07-tess-known-001",
  "method": "gpu_bls",
  "n_targets": 1000,
  "recovery_rate": 0.842,
  "median_period_error": 0.00021,
  "runtime_sec": 312.4,
  "speedup_vs_astropy": 18.7,
  "gpu": "NVIDIA RTX ...",
  "missed_targets": [
    {
      "target_id": "TIC ...",
      "known_period": 12.3,
      "reason": "low_snr_or_data_gap"
    }
  ]
}
```

### B.4.2 Markdown

論文・GitHub README向け。

### B.4.3 HTML

可視化付き。

含める図：

- periodogram
- phase-folded light curve
- recovery heatmap
- runtime scaling plot
- CPU/GPU power difference plot
- target-level summary table

### B.4.4 Parquet/CSV

大量結果保存用。

## B.5 Bの成果物

- `benchmarks/known_planets/`
- `benchmarks/injection/`
- `benchmarks/numerical/`
- `benchmarks/preprocessing/`
- `reports/*.html`
- `reports/*.md`
- `results/*.json`
- `results/*.parquet`
- `configs/*.yaml`

---

# 新規性C: GPU探索 + injection/recovery の自動評価

## C.1 狙い

GPU探索器が本当に科学的に使えるかを、人工注入信号と既知カタログで自動検証する。

速度だけではなく、

- どの深さまで検出できるか
- どの周期まで検出できるか
- どのSNRで失敗するか
- 前処理で信号を消していないか
- GPU版で検出感度が落ちていないか

を評価する。

## C.2 自前injection

任意のlight curveに人工transitを注入する。

### C.2.1 Box transit injection

最初は箱型でよい。

パラメータ：

- period
- epoch
- duration
- depth_ppm

```python
inject_box_transit(
    time,
    flux,
    period_days=5.0,
    epoch=1345.2,
    duration_hours=3.0,
    depth_ppm=500
)
```

### C.2.2 Transit model injection

次段階。

- limb darkening
- impact parameter
- planet radius ratio
- semi-major axis / stellar radius

PyTransitやbatman等と接続可能にする。

### C.2.3 Injection grid

回収率マップを作る。

```yaml
period_days: [1, 3, 10, 30, 100]
depth_ppm: [50, 100, 300, 1000]
duration_hours: [1, 2, 4, 8]
epochs: random
n_trials_per_cell: 100
```

出力：

```text
depth vs period の completeness heatmap
```

## C.3 Kepler injection products接続

Keplerのpixel-level transit injection / TCE系データを使えるようにする。

目的：

- 自前injectionだけではなく、公式に近い注入実験に対する評価を行う
- Kepler pipeline/Robovetterと比較できる
- occurrence rate研究に近い評価を行う

必要機能：

- injection table loader
- injected period/depth/duration reader
- recovered candidate matcher
- completeness calculator

## C.4 Recovery判定

### C.4.1 period match

以下をmatchとする。

- `abs(P_detected - P_true) / P_true < tolerance`
- harmonic matchを許容
  - `P_detected ≈ P_true / 2`
  - `P_detected ≈ 2 * P_true`
  - `P_detected ≈ P_true / 3`
  - `P_detected ≈ 3 * P_true`

### C.4.2 epoch match

periodが一致してもepochがズレる場合がある。

判定：

```text
phase_error < threshold
```

### C.4.3 duration/depth match

任意。

BLSではduration/depthの推定誤差は大きくなり得るため、最初は補助指標にする。

### C.4.4 detection score threshold

- SNR threshold
- BLS power threshold
- likelihood threshold
- false alarm probability

## C.5 Cの成果物

- `astrotransit_gpu.inject`
- `astrotransit_gpu.recovery`
- `astrotransit_gpu.completeness`
- `astrotransit_gpu.match`
- injection/recovery report

---

## 4. 目標ユーザー

### 4.1 第一ユーザー: 自分

最初の目的は、自分が研究として再実行できる基盤を作ること。

### 4.2 第二ユーザー: 天文学・データ科学の学生

以下を知りたい人。

- TESS/Keplerデータをどう取るか
- BLSとは何か
- 既知惑星をどう再検出するか
- GPU化でどれだけ速くなるか
- injection/recoveryとは何か

### 4.3 第三ユーザー: 研究者・OSS利用者

以下を求める人。

- 大量light curveを速く探索したい
- CPU/GPUで公平比較したい
- 前処理ごとの差を評価したい
- 新しい探索アルゴリズムを比較したい
- benchmark suiteが欲しい

---

## 5. 非目標

最初から狙わないこと。

1. 新惑星の発見を主目的にしない
2. SPOC/QLPの完全代替を狙わない
3. pixel-level photometryから始めない
4. 完全なvetting pipelineを作らない
5. 全ミッション対応を最初から狙わない
6. BLS/TLS/CETRAすべてを最初から再実装しない
7. 学術的に未検証のblack box ML分類を主軸にしない

最初の主張は、

> **既知惑星・人工注入惑星を、再現可能に、GPUで高速に再検出・評価する**

に絞る。

---

## 6. 全体アーキテクチャ

```text
AstroTransit-GPU
├── Data Layer
│   ├── Lightkurve/MAST downloader
│   ├── NASA Exoplanet Archive client
│   ├── TOI/KOI/TCE loaders
│   └── local cache
│
├── Preprocessing Layer
│   ├── quality mask
│   ├── NaN removal
│   ├── normalization
│   ├── flatten/detrending
│   └── sector stitching
│
├── Search Layer
│   ├── CPU Astropy baseline
│   ├── CPU reference implementation
│   ├── GPU BLS backend
│   └── optional CETRA/cuvarbase adapter
│
├── Validation Layer
│   ├── known planet matcher
│   ├── TOI/KOI matcher
│   ├── injection engine
│   ├── recovery evaluator
│   └── numerical comparator
│
├── Benchmark Layer
│   ├── runtime benchmark
│   ├── scaling benchmark
│   ├── accuracy benchmark
│   ├── preprocessing benchmark
│   └── hardware profiler
│
└── Report Layer
    ├── JSON output
    ├── Markdown output
    ├── HTML dashboard
    └── plots
```

---

## 7. 推奨リポジトリ構成

```text
astrotransit-gpu/
├── README.md
├── pyproject.toml
├── environment.yml
├── Dockerfile
├── LICENSE
├── CITATION.cff
│
├── src/
│   └── astrotransit_gpu/
│       ├── __init__.py
│       ├── cli.py
│       │
│       ├── data/
│       │   ├── lightkurve_client.py
│       │   ├── mast_cache.py
│       │   ├── exoplanet_archive.py
│       │   ├── toi.py
│       │   ├── koi.py
│       │   ├── tce.py
│       │   └── schemas.py
│       │
│       ├── preprocess/
│       │   ├── quality.py
│       │   ├── normalize.py
│       │   ├── flatten.py
│       │   ├── stitch.py
│       │   └── masks.py
│       │
│       ├── search/
│       │   ├── astropy_bls.py
│       │   ├── cpu_reference_bls.py
│       │   ├── gpu_bls.py
│       │   ├── kernels/
│       │   │   ├── bls.cu
│       │   │   ├── reduction.cu
│       │   │   └── topk.cu
│       │   └── adapters/
│       │       ├── cuvarbase_adapter.py
│       │       └── cetra_adapter.py
│       │
│       ├── inject/
│       │   ├── box.py
│       │   ├── transit_model.py
│       │   ├── grid.py
│       │   └── kepler_injection.py
│       │
│       ├── validate/
│       │   ├── match.py
│       │   ├── known_planets.py
│       │   ├── recovery.py
│       │   ├── numerical.py
│       │   └── metrics.py
│       │
│       ├── benchmark/
│       │   ├── runtime.py
│       │   ├── scaling.py
│       │   ├── preprocessing.py
│       │   └── profiler.py
│       │
│       └── report/
│           ├── json_report.py
│           ├── markdown_report.py
│           ├── html_report.py
│           └── plots.py
│
├── configs/
│   ├── tess_known_100.yaml
│   ├── kepler_known_100.yaml
│   ├── injection_grid_small.yaml
│   └── numerical_compare.yaml
│
├── benchmarks/
│   ├── known_planets/
│   ├── injection/
│   ├── numerical/
│   └── preprocessing/
│
├── notebooks/
│   ├── 01_known_planet_redetection.ipynb
│   ├── 02_injection_recovery.ipynb
│   ├── 03_cpu_gpu_comparison.ipynb
│   └── 04_preprocessing_effects.ipynb
│
├── tests/
│   ├── test_cpu_reference.py
│   ├── test_gpu_bls_small.py
│   ├── test_matching.py
│   ├── test_injection.py
│   └── test_reproducibility.py
│
├── docs/
│   ├── design.md
│   ├── algorithm.md
│   ├── validation.md
│   ├── benchmark_protocol.md
│   └── references.md
│
└── reports/
    └── .gitkeep
```

---

## 8. CLI仕様

## 8.1 既知惑星再検出

```bash
astrotransit-gpu known \
  --mission tess \
  --catalog toi \
  --n-targets 100 \
  --method gpu-bls \
  --config configs/tess_known_100.yaml \
  --out reports/tess_known_100
```

## 8.2 CPU/GPU比較

```bash
astrotransit-gpu compare \
  --target "TIC 261136679" \
  --cpu astropy \
  --gpu cuda \
  --period-min 0.5 \
  --period-max 30 \
  --out reports/compare_tic_261136679
```

## 8.3 自前injection/recovery

```bash
astrotransit-gpu inject-run \
  --target "Kepler-10" \
  --period-days 3,10,30 \
  --depth-ppm 100,300,1000 \
  --duration-hours 1,3,6 \
  --method gpu-bls \
  --n-trials 50 \
  --out reports/injection_kepler10
```

## 8.4 Kepler injection benchmark

```bash
astrotransit-gpu kepler-injection \
  --release dr25 \
  --n-targets 1000 \
  --method gpu-bls \
  --out reports/kepler_injection_dr25_gpu
```

## 8.5 前処理比較

```bash
astrotransit-gpu preprocess-benchmark \
  --targets targets/toi_small.csv \
  --methods pdcsap,flatten,qlp,tess-spoc \
  --search gpu-bls \
  --out reports/preprocess_benchmark
```

---

## 9. Python API仕様

```python
from astrotransit_gpu import TransitSearchPipeline

pipeline = TransitSearchPipeline(
    mission="tess",
    search_method="gpu_bls",
    period_range=(0.5, 30.0),
    durations_hours=[1, 2, 4, 8],
)

result = pipeline.run_target("TIC 261136679")

print(result.best_period)
print(result.best_epoch)
print(result.best_duration)
print(result.snr)
```

## 9.1 既知惑星照合

```python
from astrotransit_gpu.validate import match_known_planet

match = match_known_planet(
    detected_period=result.best_period,
    detected_epoch=result.best_epoch,
    catalog_period=known.period,
    catalog_epoch=known.epoch,
    allow_harmonics=True,
)
```

## 9.2 injection

```python
from astrotransit_gpu.inject import inject_box_transit
from astrotransit_gpu.validate import evaluate_recovery

time2, flux2, truth = inject_box_transit(
    time,
    flux,
    period_days=10.0,
    duration_hours=3.0,
    depth_ppm=300,
    seed=42,
)

detected = pipeline.search(time2, flux2)
recovery = evaluate_recovery(detected, truth)
```

---

## 10. データ設計

## 10.1 TargetRecord

```python
@dataclass
class TargetRecord:
    mission: str
    target_id: str
    source_catalog: str
    ra: float | None
    dec: float | None
    sectors_or_quarters: list[int]
    known_period_days: float | None
    known_epoch_bjd: float | None
    known_duration_hours: float | None
    known_depth_ppm: float | None
    disposition: str | None
```

## 10.2 LightCurveRecord

```python
@dataclass
class LightCurveRecord:
    target_id: str
    mission: str
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None
    quality: np.ndarray | None
    sector_or_quarter: np.ndarray | None
    flux_type: str
    provenance: str
```

## 10.3 SearchResult

```python
@dataclass
class SearchResult:
    target_id: str
    method: str
    best_period_days: float
    best_epoch_bjd: float
    best_duration_hours: float
    best_depth_ppm: float | None
    best_score: float
    score_type: str
    top_candidates: list[Candidate]
    runtime_sec: float
    hardware: dict
    config_hash: str
```

## 10.4 RecoveryResult

```python
@dataclass
class RecoveryResult:
    target_id: str
    truth_period_days: float
    detected_period_days: float | None
    recovered: bool
    match_type: str
    period_error_frac: float | None
    phase_error: float | None
    depth_ppm: float
    duration_hours: float
    snr: float | None
```

---

## 11. 評価指標

## 11.1 速度系

| 指標 | 意味 |
|---|---|
| runtime_sec | 総実行時間 |
| light_curves_per_sec | 1秒あたり処理本数 |
| period_duration_evals_per_sec | 探索評価回数/秒 |
| gpu_memory_peak_mb | GPU最大メモリ |
| speedup_vs_astropy | Astropy比 |
| speedup_vs_cpu_reference | 自作CPU比 |
| energy_per_light_curve | 可能なら消費電力あたり処理量 |

## 11.2 正しさ系

| 指標 | 意味 |
|---|---|
| period_error_frac | 周期誤差 |
| epoch_phase_error | 位相誤差 |
| duration_error_frac | duration誤差 |
| depth_error_frac | depth誤差 |
| topk_contains_truth | top-k候補に正解があるか |
| harmonic_match_rate | harmonicに落ちた割合 |
| cpu_gpu_power_rmse | CPU/GPU periodogram差分 |

## 11.3 科学系

| 指標 | 意味 |
|---|---|
| recovery_rate | 既知/注入惑星の回収率 |
| completeness | depth/periodごとの検出率 |
| false_alarm_rate | 偽陽性率 |
| reliability | 検出候補のうち真陽性割合 |
| missed_target_analysis | 見逃し理由の分類 |
| preprocessing_sensitivity | 前処理ごとの検出率差 |

---

## 12. マッチング仕様

## 12.1 Period Match

```python
def period_match(p_detected, p_true, tolerance=0.01):
    return abs(p_detected - p_true) / p_true < tolerance
```

## 12.2 Harmonic Match

許容するharmonic。

```text
1/3 P
1/2 P
1 P
2 P
3 P
```

実装：

```python
ratios = [1/3, 1/2, 1, 2, 3]
best = min(abs(p_detected - p_true*r)/(p_true*r) for r in ratios)
```

## 12.3 Phase Match

```python
phase_error = min(
    abs((epoch_detected - epoch_true) % period),
    period - abs((epoch_detected - epoch_true) % period)
) / period
```

## 12.4 Recovery判定

最初の標準判定：

```text
recovered = period_match_or_harmonic
```

厳密判定：

```text
recovered_strict = period_match_or_harmonic and phase_error < 0.1
```

---

## 13. GPU BLS実装方針

## 13.1 基本方針

最初はAstropy BLSと同一の完全再現を目指すのではなく、

1. 小さい合成データでCPU referenceと一致
2. Astropyとbest periodが一致
3. known planet benchmarkで同程度のrecovery
4. injection benchmarkで同程度のcompleteness

を段階的に満たす。

## 13.2 メモリ配置

候補：

```text
time:  [n_stars, n_time]
flux:  [n_stars, n_time]
mask:  [n_stars, n_time]
periods: [n_periods]
durations: [n_durations]
scores: [n_stars, n_periods, n_durations]
```

ただし、`scores` 全量は巨大になるため、最終的には top-k のみ保持する。

## 13.3 Kernel分割

初期実装：

1. `compute_bls_scores_kernel`
2. `topk_candidates_kernel`

最適化後：

1. `preprocess_and_score_kernel`
2. `block_reduce_topk_kernel`
3. `merge_topk_kernel`

## 13.4 Top-k方針

全periodogramを保存するとメモリが重い。  
標準では top-k のみ返す。

```yaml
search:
  return_full_periodogram: false
  top_k: 10
```

研究・デバッグ時のみ全periodogramを保存。

## 13.5 精度

最初はfloat32。  
正しさ検証ではfloat64 CPU基準と比較する。

比較対象：

- float32 GPU
- float64 GPU可能なら
- mixed precision
- reduction順序による差分

---

## 14. 前処理設計

## 14.1 最初の標準前処理

```text
1. quality flag filtering
2. NaN removal
3. median normalization
4. optional flatten
5. optional sigma clipping
6. sector stitching
```

## 14.2 注意点

前処理は信号を消す可能性がある。  
特に長周期・長duration・浅いtransitで問題になる。

必ず前処理比較ベンチを設ける。

## 14.3 最初は既存light curveを使う

最初からTPF/FFIからphotometryしない。  
使う順番：

1. SPOC PDCSAP
2. QLP
3. TESS-SPOC
4. eleanor
5. 必要ならTPF/FFI

---

## 15. レポート仕様

## 15.1 Known Planet Report

含める内容：

- 実行設定
- データソース
- 対象数
- 回収率
- 速度
- 既知値との差
- 見逃し一覧
- harmonic一覧
- 代表例のperiodogram
- 代表例のfolded light curve

## 15.2 Injection Report

含める内容：

- injection grid
- completeness heatmap
- depth別回収率
- period別回収率
- duration別回収率
- false alarm estimate
- CPU/GPU比較
- 失敗例

## 15.3 Numerical Report

含める内容：

- CPU/GPU best candidate比較
- periodogram差分
- top-k overlap
- float32/float64差分
- small synthetic test結果

---

## 16. 最初のMVP

## 16.1 MVP名

**MVP-0: Known Planet Re-detection on 10 Targets**

## 16.2 目的

TESS/Keplerの公開光度曲線から、10個の既知惑星をBLSで再検出し、CPU/GPU比較の最小基盤を作る。

## 16.3 入力

- 10個の明るく有名なTESS/Kepler target
- Lightkurveで取得したPDCSAP/SAP light curve
- NASA Exoplanet Archiveから取得したperiod/epoch/duration

## 16.4 実装

- Lightkurve downloader
- Astropy BLS baseline
- 自作CPU reference
- CuPy or PyTorch GPU prototype
- known planet matcher
- JSON report

## 16.5 成功条件

- 10個中7個以上で既知周期またはharmonicを再検出
- GPU版がCPU referenceと同じbest periodを返す小テストを通過
- JSON/Markdownレポートが生成される
- seed/config固定で再実行可能

---

## 17. Phase計画

# Phase 0: 調査・設計固定

期間: 3〜5日

作業：

- LightkurveでTESS/Keplerデータ取得を確認
- NASA Exoplanet Archive TAPの取得コードを書く
- Astropy BLSを10 targetで動かす
- config schemaを固定
- result schemaを固定

成果物：

- `notebooks/01_known_planet_redetection.ipynb`
- `configs/tess_known_10.yaml`
- `docs/benchmark_protocol.md`

---

# Phase 1: CPU基準と検証基盤

期間: 1〜2週間

作業：

- Astropy BLS wrapper
- 自作CPU reference small version
- known planet matcher
- harmonic matcher
- JSON/Markdown report
- 10〜100 target benchmark

成果物：

- CPU baseline report
- known planet recovery table
- missed target analysis

成功条件：

- Lightkurve + Astropyで既知惑星再検出が再現できる
- Archive照合が自動化される

---

# Phase 2: GPU prototype

期間: 2〜4週間

作業：

- CuPy/PyTorchでBLS風探索をGPU化
- 小さいsynthetic dataでCPU referenceと比較
- 10 targetでGPU探索
- runtime benchmark

成果物：

- `gpu_bls.py`
- numerical report
- speedup report

成功条件：

- small syntheticでCPU referenceと一致
- known targetでCPUと同じまたは近いcandidateを返す
- CPUより高速

---

# Phase 3: CUDA backend

期間: 1〜2か月

作業：

- CUDA C++ kernel
- pybind11/CuPy RawKernel
- batched light curve対応
- top-k candidate抽出
- GPU profiling
- memory最適化

成果物：

- `kernels/bls.cu`
- benchmark table
- profiler report

成功条件：

- 100〜1000 light curvesで実用的なspeedup
- メモリ不足なく処理可能
- CPU/GPU差分が許容範囲

---

# Phase 4: Injection/recovery

期間: 1〜2か月

作業：

- 自前box injection
- transit model injection
- injection grid
- recovery evaluator
- completeness heatmap
- Kepler injection products loaderの初期対応

成果物：

- injection report
- recovery heatmap
- completeness table

成功条件：

- depth/period/durationごとの回収率が出る
- CPU/GPUで回収率比較ができる

---

# Phase 5: 統合ベンチマーク・OSS化

期間: 1〜2か月

作業：

- Docker
- CI
- tests
- docs
- HTML report
- ASCL向け整備
- example notebooks
- paper draft

成果物：

- v0.1.0 release
- documentation site
- reproducible benchmark
- preprint draft

---

## 18. 論文化の方向

## 18.1 論文タイトル案

### 案1

**AstroTransit-GPU: A Reproducible GPU-Accelerated Transit Search Benchmark for Public TESS and Kepler Light Curves**

### 案2

**A Reproducible GPU Framework for Transit Recovery and Injection Benchmarks in TESS and Kepler Light Curves**

### 案3

**Fast and Verifiable Exoplanet Transit Search on GPUs with Automated Known-Planet and Injection-Recovery Evaluation**

## 18.2 主張

主張は以下の3本柱。

1. **GPU高速化**
   - Astropy/CPU基準より高速
   - 大量light curve batchで効果が大きい

2. **再現可能ベンチマーク**
   - config固定
   - data provenance保存
   - Docker/CI
   - JSON/HTMLレポート

3. **科学的正しさ確認**
   - 既知惑星再検出
   - TOI/KOI/TCE照合
   - injection/recovery
   - CPU/GPU数値差分

## 18.3 投稿先候補

- Astronomy and Computing
- Journal of Open Source Software
- Research Notes of the AAS
- PASP
- ApJS software paper
- ASCL登録
- arXiv astro-ph.IM

最初は **JOSS + ASCL + arXiv** が現実的。

---

## 19. リスクと対策

## 19.1 既存GPUツールとの差別化が弱い

### リスク

cuvarbaseやCETRAが既にあるため、単なるGPU BLSでは新規性が弱い。

### 対策

主張を「GPU BLS単体」ではなく、

- 再現可能ベンチ
- 公式カタログ照合
- injection/recovery自動化
- CPU/GPU差分評価
- 前処理比較

に置く。

## 19.2 前処理で結果が変わりすぎる

### リスク

探索器の性能ではなく前処理の差を測ってしまう。

### 対策

- 前処理をconfig固定
- raw/PDCSAP/flatten等を分けて報告
- 前処理比較を独立ベンチにする
- injectionを前処理前/後の両方で評価

## 19.3 GPU版がAstropyと完全一致しない

### リスク

BLS実装細部の違いで結果がズレる。

### 対策

- small syntheticでは自作CPU referenceと一致させる
- Astropyとは候補一致/回収率で比較する
- periodogram全体一致を必須にしない
- 差分を明示的にレポートする

## 19.4 データ取得が不安定

### リスク

MASTやArchiveへのアクセスで再現性が崩れる。

### 対策

- local cache
- manifest保存
- downloaded file hash
- small benchmark datasetを固定
- optional offline mode

## 19.5 計算環境差

### リスク

GPU種類で速度が大きく変わる。

### 対策

- hardware metadata保存
- CPU baselineも必ず出す
- relative speedupを重視
- Docker/Apptainer提供

---

## 20. 最優先で作るファイル

最初に作る順。

```text
1. pyproject.toml
2. src/astrotransit_gpu/data/exoplanet_archive.py
3. src/astrotransit_gpu/data/lightkurve_client.py
4. src/astrotransit_gpu/search/astropy_bls.py
5. src/astrotransit_gpu/validate/match.py
6. src/astrotransit_gpu/report/json_report.py
7. notebooks/01_known_planet_redetection.ipynb
8. configs/tess_known_10.yaml
9. src/astrotransit_gpu/search/gpu_bls.py
10. src/astrotransit_gpu/inject/box.py
```

---

## 21. 最初の実験セット

## 21.1 Known Planet Small

目的：

- 既知惑星再検出の流れを作る

対象数：

- 10

方法：

- Lightkurve + Astropy BLS
- Archive照合
- Markdown/JSON出力

## 21.2 Synthetic Small

目的：

- CPU/GPU一致確認

データ：

- 正弦なし
- noiseあり
- box transitあり
- fixed seed

判定：

- best period一致
- top-k一致
- score差分

## 21.3 Injection Small

目的：

- injection/recovery基盤を作る

条件：

- period: 3, 10, 30 days
- depth: 100, 300, 1000 ppm
- duration: 2, 4 hours
- n_trials: 10

---

## 22. 予想される最初のREADME構成

```markdown
# AstroTransit-GPU

GPU-accelerated, reproducible transit search and recovery benchmarks for public TESS and Kepler light curves.

## Features

- Download TESS/Kepler light curves via Lightkurve
- Run CPU baseline with Astropy BLS
- Run GPU BLS backend
- Validate against NASA Exoplanet Archive TOI/KOI/confirmed planets
- Run injection/recovery benchmarks
- Generate JSON/Markdown/HTML reports

## Quickstart

astrotransit-gpu known --mission tess --catalog toi --n-targets 10

## Why this exists

Existing tools provide data access, CPU BLS, GPU search, and catalogs separately. AstroTransit-GPU integrates them into a reproducible benchmark framework.

## Status

Experimental.
```

---

## 23. 最終的な到達点

理想的なv1.0は以下。

```bash
astrotransit-gpu run configs/paper_tess_kepler_benchmark.yaml
```

これだけで、

1. TESS/Kepler target listを取得
2. light curveをダウンロード/cache
3. 前処理
4. CPU BLS
5. GPU BLS
6. 既知惑星照合
7. injection/recovery
8. CPU/GPU差分
9. HTML/Markdown/JSONレポート生成
10. 論文用図表生成

まで完了する。

---

## 24. まとめ

作るべきものは、単なるGPU実装ではない。

作るべきものは、

# **AstroTransit-GPU: 検証可能なGPU高速トランジット探索基盤**

である。

採用する新規性は以下。

## A. GPU高速化

BLS互換探索をGPUで高速化し、大量TESS/Kepler light curveを高速処理する。

## B. 再現可能ベンチマーク

データ、前処理、period grid、duration grid、探索器、hardware、結果をすべてconfigとmanifestで固定し、誰でも再実行できるようにする。

## C. GPU探索 + injection/recovery

既知惑星だけでなく、自前injectionとKepler injection productsを使って、GPU探索器の科学的な回収率を評価する。

この3つを統合すると、既存ツールとの差別化は明確になる。

> 既存ツールは「部品」として強い。  
> AstroTransit-GPU は「正しさを確認できる高速探索実験基盤」として価値を出す。

---

## 参考情報・主要ソース

- Lightkurve tutorial: Identifying transiting exoplanet signals in a light curve  
  https://lightkurve.github.io/lightkurve/tutorials/3-science-examples/exoplanets-identifying-transiting-planet-signals.html

- Astropy BoxLeastSquares API  
  https://docs.astropy.org/en/stable/api/astropy.timeseries.BoxLeastSquares.html

- Astropy BLS documentation  
  https://docs.astropy.org/en/stable/timeseries/bls.html

- TESS Data Products  
  https://heasarc.gsfc.nasa.gov/docs/tess/data-products.html

- NASA Exoplanet Archive TAP  
  https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html

- TESS Objects of Interest column definitions  
  https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html

- Kepler KOI documentation  
  https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html

- Kepler TCE documentation  
  https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_TCE_docs.html

- Kepler Data Products Overview  
  https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_Data_Products_Overview.html

- CETRA: a fast, sensitive exoplanet transit detection algorithm implemented for GPUs  
  https://academic.oup.com/mnras/article/539/1/297/8099934

- PyTransit documentation  
  https://pytransit.readthedocs.io/en/latest/models.html

- TESS Quick-Look Pipeline update  
  https://tess.mit.edu/qlp-update/
