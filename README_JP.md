# AstroTransit-GPU (v1.3.0)

[![CI](https://github.com/ubunbun18/Astro_Transit_GPU/actions/workflows/ci.yml/badge.svg)](https://github.com/ubunbun18/Astro_Transit_GPU/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-1.3.0-blue.svg)](./CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**AstroTransit-GPU** は、TESS/Kepler 光度曲線に対するトランジット探索（Box Least Squares）を CUDA で高速化し、Astropy との数値的一致を保証しつつ、再現可能なベンチマークと注入実験（Injection/Recovery）を可能にする研究者向けの GPU 探索基盤です。

## 🌟 特徴

- **超高速かつ正確**: Astropy比で100倍以上のスループットを達成しつつ、高い科学的妥当性を維持。
- **Apex Predator (V39)**: Blackwellアーキテクチャに最適化された最新の堅牢カーネル。**219億組の探索を約9.4分**で完了（~3,900万組/秒）し、重み付きSNRに対応。
- **科学的バリデーション済み**: NASA TOIカタログとの照合により、Sector 1 FFIデータで **38.75%の回収率** を実証。
- **サーベイスケール・パイプライン**: `SectorCache` システムによりI/Oボトルネックを排除し、セクター全件（1.6万天体）を **45秒以内** にスクリーニング。
- **Astropy 互換 API**: 既存のワークフローに組み込みやすいオブジェクト指向設計。
- **堅牢な自動運用**: 破損した FITS キャッシュの自動検知・削除・リトライ機能を搭載。
- **実データ対応**: `flux_err`（重み付き解析）および `float64`（倍精度）をフルサポート。
- **高い再現性**: YAML 設定ファイル、シード固定、統計的計測（Median/P95）による信頼性の担保。

## 🚀 Blackwell Singularity (V37)

TESS QLP 全天サーベイのような超大規模スクリーニング向けに、**V37 "Apex Predator"** エンジンを提供しています。このエンジンは NVIDIA Blackwell (および RTX 40 シリーズ) GPU のハードウェア特性を最大限に引き出すよう設計されています。

### 主な最適化:
- **Winner-Take-All 出力**: スピードを優先し、全周期のスコア（スペクトル）ではなく、**各ターゲットの最高スコア一点のみ**を返します。これにより、メモリ帯域のボトルネックを解消しています。
- **Zero-Div ロジック**: 高レイテンシな除算器を避け、交差乗算による比較により演算スループットを極大化。
- **Zero-Spill SMEM**: ビニニング計算をバンク最適化された共有メモリ (SMEM) に展開し、レジスタ溢れを完全に排除。
- **Warp-Parallel Scan**: ワープシャッフルを用いた並列プレフィックス和により、集計フェーズのボトルネックを解消。

### 使用方法 (CLI):
```bash
python -m astrotransit_gpu screen-sector \
  --cache-dir data/sector1_cache \
  --out outputs/bench_219331_v39.csv \
  --n-periods 100000 \
  --blackwell
```

### 使用方法 (Python API):
```python
periods = np.linspace(0.5, 20.0, 100000)
durations = np.linspace(0.01, 0.2, 5)

screener = GpuScreener(periods, durations, n_bins=128)
results = screener.screen_sector_vbls(data, use_blackwell=True)
```

## 🚀 インストール

CPU 環境でもインストール可能ですが、GPU 加速を利用するには CUDA 対応の CuPy が必要です。

```bash
# 開発モードでインストール (推奨)
git clone https://github.com/ubunbun18/Astro_Transit_GPU.git
cd Astro_Transit_GPU
pip install -e ".[cuda12,benchmark]"
```

## 🛠️ クイックスタート (Python API)

```python
from astrotransit_gpu import BoxLeastSquaresGPU
import numpy as np

# データの準備 (TESS/Kepler等)
t = np.linspace(0, 10, 5000)
y = np.ones_like(t)  # 光度データ
dy = np.ones_like(t) * 0.001 # 誤差 (オプション)

# モデルの初期化 (Astropy互換)
model = BoxLeastSquaresGPU(t, y, dy=dy)

# 探索の実行
periods = np.linspace(0.5, 20.0, 10000)
durations = [0.05, 0.1, 0.15]
results = model.power(periods, durations, n_bins=500)

print(f"Best Period: {results.best_period:.4f} days")
print(f"Best Power (SNR): {results.best_power:.2f}")
```

## 💻 CLI コマンド

| コマンド | 説明 |
| :--- | :--- |
| `check` | GPU の可用性と CUDA 環境の診断。 |
| `compare` | CPU と GPU の数値パリティ・性能比較。`--preset` を使用可能。 |
| `inject` | 注入実験を実行し、回収率マップ（Recovery Heatmap）を生成。 |
| `benchmark` | 設定ファイル（YAML）から再現可能な検証レポートを自動生成。 |
| `search` | 単一ターゲットに対する高速探索と結果表示。 |
| `batch` | ターゲットリストに基づく一括並列解析。 |
| `build-cache` | 大量の FITS ファイルを高速な単一バイナリ形式に集約。 |
| `screen-sector` | 集約キャッシュを用いた超高速セクター一括スクリーニング。 |

詳細な引数や実行例については、[CLI 完全リファレンス (docs/CLI_REFERENCE_JP.md)](./docs/CLI_REFERENCE_JP.md) を参照してください。

## 📊 ベンチマークと検証

### 公式検証パイプライン (V39 Reproducibility)
V39 カーネルの科学的妥当性を再現するための標準手順が用意されています。

```bash
# 公式検証スクリプトの実行
astrotransit-gpu validate --results outputs/bench_219331_v39.csv
```

#### 科学的判定基準 (Standard Criteria):
検証に使用されるデフォルトの設定（`configs/validation_v39.yaml`）は以下の通りです：

- **SNR 閾値**: `7.1` (TESS 標準の有意性レベル)
- **周期許容誤差 (p_tol)**: `0.01` (真の周期の 1% 以内)
- **T0 許容誤差 (t0_tol)**: `0.5` 日
- **完備性対象**: 観測期間内で 2 回以上のトランジットが物理的に期待される天体（P < 13.7d）

詳細な検証手法と最新のベンチマーク結果については、[BENCHMARK_REPORT_JP.md](./BENCHMARK_REPORT_JP.md) および [MASSIVE_VALIDATION_REPORT_JP.md](./reports/MASSIVE_VALIDATION_REPORT_JP.md) を参照してください。

## 📖 制限事項と注意点

- 現バージョンの GPU バックエンドは位相ビン詰め（Phase Binning）アルゴリズムを使用しています。
- デフォルトの精度は `float32` です。超長期間のデータで精度が必要な場合は `float64` を指定してください。
- 本パッケージは候補の高速スクリーニングを目的としており、最終的な MCMC フィッティング機能は含みません。

## 📄 引用 (Citation)

研究で本ソフトウェアを使用された場合は、[CITATION.cff](./CITATION.cff) を参照して引用してください。

## ⚖️ ライセンス

MIT License - 詳細は [LICENSE](./LICENSE) を参照してください。
