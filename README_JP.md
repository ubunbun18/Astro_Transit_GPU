# AstroTransit-GPU (v1.0.0)

[![CI](https://github.com/ubunbun18/Astro_Transit_GPU/actions/workflows/ci.yml/badge.svg)](https://github.com/ubunbun18/Astro_Transit_GPU/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](./CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**AstroTransit-GPU** は、TESS/Kepler 光度曲線に対するトランジット探索（Box Least Squares）を CUDA で高速化し、Astropy との数値的一致を保証しつつ、再現可能なベンチマークと注入実験（Injection/Recovery）を可能にする研究者向けの GPU 探索基盤です。

## 🌟 特徴

- **高速かつ正確**: CUDA による並列化で Astropy 比 100 倍以上のスループットを達成しつつ、スペクトル全体の極めて高い相関（[Correlation > 0.95](./BENCHMARK_REPORT_JP.md#数値一致性の検証)）を維持。
- **Survey Scale パイプライン**: Sector 単位の **Flat Binary Cache** 構築により、ディスク I/O 待ちを完全に解消。
- **超高速スクリーニング**: 1.6 万天体に対しても **10 万周期探索を約 30 分**（約 9 天体/秒）で完遂。
- **Astropy 互換 API**: 既存のワークフローに組み込みやすいオブジェクト指向設計。
- **堅牢な自動運用**: 破損した FITS キャッシュの自動検知・削除・リトライ機能を搭載。
- **実データ対応**: `flux_err`（重み付き解析）および `float64`（倍精度）をフルサポート。
- **高い再現性**: YAML 設定ファイル、シード固定、統計的計測（Median/P95）による信頼性の担保。

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

詳細な検証手法と最新のベンチマーク結果については、[BENCHMARK_REPORT_JP.md](./BENCHMARK_REPORT_JP.md) を参照してください。

## 📖 制限事項と注意点

- 現バージョンの GPU バックエンドは位相ビン詰め（Phase Binning）アルゴリズムを使用しています。
- デフォルトの精度は `float32` です。超長期間のデータで精度が必要な場合は `float64` を指定してください。
- 本パッケージは候補の高速スクリーニングを目的としており、最終的な MCMC フィッティング機能は含みません。

## 📄 引用 (Citation)

研究で本ソフトウェアを使用された場合は、[CITATION.cff](./CITATION.cff) を参照して引用してください。

## ⚖️ ライセンス

MIT License - 詳細は [LICENSE](./LICENSE) を参照してください。
