# AstroTransit-GPU 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**AstroTransit-GPU** は、TESS や Kepler などの時系列光度データから惑星トランジットを探索するための、CUDA 加速プラットフォームです。独自の CUDA カーネルによる並列演算と非同期処理パイプラインにより、大規模な系外惑星サーベイにおける探索スループットを最大化します。

## 🔬 技術的特徴

-   **並列 BLS カーネル**: 
    - **周期タイリング**: 1回のメモリ読み込みで複数の周期を並列処理し、スループットを向上。
    - **並列累積和探索**: スレッド間通信を利用した高速なスライディングウィンドウ探索。
-   **非同期データ処理**:
    - `ProcessPoolExecutor` による CPU 前処理（ダウンロード・クリーニング）と GPU 演算のオーバーラップ。
-   **数値的一致の検証**:
    - 業界標準の `astropy.timeseries.BoxLeastSquares` と同一の探索グリッド上で比較検証を行い、科学的な整合性を確保。

## 📊 パフォーマンスと信頼性

AstroTransit-GPU は、探索グリッドが高密度になるほど、CPU (Astropy) に対するスループットの優位性が拡大します。

| 探索規模 | 周期グリッド数 | 加速倍率 (CPU比) |
| :--- | :--- | :--- |
| **Standard** | 5,000 | 約 7.5倍 |
| **Large** | 100,000 | 約 79倍 |
| **Extreme** | 1,000,000 | **130倍以上** |

> [!NOTE]
> 詳細な測定条件、数値的な一致度（精度）、および再現コマンドについては、[BENCHMARK_REPORT_JP.md](./BENCHMARK_REPORT_JP.md) を参照してください。

## 🚀 インストール

```bash
git clone https://github.com/ubunbun18/AstroTransit-GPU.git
cd AstroTransit-GPU
pip install .
```

## 🛠️ CLI コマンド

| コマンド | 説明 |
| :--- | :--- |
| `check` | GPU の可用性と CUDA 環境を診断。 |
| `compare` | CPU と GPU の速度・精度を直接比較。 |
| `known` | 特定の既知ターゲットに対する探索とレポート生成。 |
| `batch` | NASA カタログからターゲットを一括取得し、非同期並列解析。 |
| `inject-run` | 信号注入実験を行い、回収率マップを生成。 |
| `run-config` | YAML 設定ファイルに基づいた実験の実行。 |

詳細なオプションと使用例は [BENCHMARK_REPORT_JP.md](./BENCHMARK_REPORT_JP.md#2-cli-完全リファレンス) に記載されています。

## 📋 性能の再現方法

お使いの環境で性能を測定するには、以下のコマンドを実行してください：

```bash
# 標準解像度での比較
astrotransit-gpu compare --preset standard
```

## 📝 ライセンス

MIT License。詳細は `LICENSE` ファイルを参照してください。

---
*Scaling Exoplanet Discovery with Reliability.*
