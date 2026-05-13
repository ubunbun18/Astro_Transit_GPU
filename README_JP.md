# AstroTransit-GPU (v1.4.0)

[![CI](https://github.com/ubunbun18/Astro_Transit_GPU/actions/workflows/ci.yml/badge.svg)](https://github.com/ubunbun18/Astro_Transit_GPU/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-1.4.0-blue.svg)](./CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**AstroTransit-GPU** は、研究者向けのハイエンド系外惑星トランジット探索プラットフォームです。カスタム CUDA カーネルによる高速計算と、Astropy との数値的一致（実測: 相関係数 0.999997）を両立させています。

---

## 🌟 主な特徴

| 機能 | 内容 |
| :--- | :--- |
| **🚀 高速スクリーニング** | 1.6万天体（TESS 1セクター）を **約 73秒** で完了 (V41, 5,000周期, 実測)。 |
| **🎯 高い再現性** | Astropy `BoxLeastSquares` との相関係数 **0.999997**、ΔT0 **1.43e-08日** を実証 (V42)。 |
| **🛠️ デュアルエンジン** | 「高速スクリーニング (V41)」と「厳密な学術検証 (V42)」を用途に合わせて選択可能。 |
| **📦 サーベイ特化** | 独自の集約バイナリキャッシュにより、ディスク I/O のボトルネックを排除。 |
| **🧪 実証済み** | NASA TOI カタログに対して **38.75% の回収率** を検証 (Sector 1 FFI)。 |
| **💻 モダンな API** | Astropy 互換の `BoxLeastSquaresGPU` クラスにより、既存コードに即導入可能。 |

---

## 🏎️ デュアルエンジン・アーキテクチャ

v1.4.0 では、トランジット探索のワークフロー全体をカバーする 2 つのエンジンを提供します。

### 1. Fast Engine (V41) — *スクリーニング用*
大規模サーベイの初期スクリーニングに特化。NVIDIA Blackwell / Ada アーキテクチャに最適化。
- **実測スループット**: **234 LC/s** (5,000周期, 15,000データ点)。
- **最適化**: Branchless 境界処理、代数的 FDIV 削減、Warp Occupancy 100%。
- **注意**: 並列アトミック演算を使用するため、実行毎に微小な数値的変動あり (Drift: 5.72e-06)。

### 2. Parity Engine (V42) — *学術検証用*
科学的再現性を最優先に設計。CPU と同一の計算順序でビットレベルの一致を実現。
- **実測精度**: Astropy との相関係数 **0.999997**、数値的ドリフト **0.000000**（完全決定論的）。
- **実測スループット**: **48 LC/s** (5,000周期, 15,000データ点)。
- **用途**: 最終候補の科学的検証、論文用データの生成。

---

## 🚀 クイックスタート

### Python API
```python
from astrotransit_gpu import BoxLeastSquaresGPU

# インスタンス化 (time, flux, flux_err)
model = BoxLeastSquaresGPU(t, y, dy=dy)

# 【高速モード】初期探索用 (デフォルト)
res_fast = model.power(periods, durations, method="fast")

# 【パリティモード】Astropy 一致・最終検証用
res_exact = model.power(periods, durations, method="parity")
```

### コマンドライン (CLI)
```bash
# GPU 診断
astrotransit-gpu check

# 特定の天体を「パリティモード」で探索
astrotransit-gpu search --target "TIC 261136679" --method parity

# セクター全体の高速スクリーニング
astrotransit-gpu screen-sector --cache-dir ./data_cache --n-periods 5000
```

---

## 📊 実測ベンチマーク (v1.4.0)

**計測機材**: AMD Ryzen 7 9700X + NVIDIA GeForce RTX 5060 Ti (Blackwell)
**計測条件**: 5,000 周期、15,000 データ点/天体

| カーネル | スループット (LC/s) | 加速倍率 (vs CPU) | 相関係数 (vs Astropy) |
| :--- | :--- | :--- | :--- |
| CPU (Astropy) | 3.97 | 1.0x | 1.000000 |
| **V41 (Fast)** | **234.07** | **58.9倍** | 0.967864 |
| **V42 (Parity)** | **48.05** | **12.1倍** | **0.999997** |

**TESS 1セクター (15,881天体) の実測完了時間**:
- V41 (Fast): **72.78 秒 (1.21 分)**
- V42 (Parity): **362.95 秒 (6.05 分)**
- CPU (Astropy): **約 4,000 秒 (66.67 分)** (3.97 LC/s より換算)

---

## 📄 ドキュメント (日本語)
- [CLI リファレンス](./docs/CLI_REFERENCE_JP.md)
- [カーネル選択ガイド (V41 vs V42)](./docs/KERNEL_GUIDE_V39_V42_JP.md)
- [ベンチマークレポート V3](./BENCHMARK_REPORT_V3_JP.md)
- [セクターキャッシュ設計](./docs/SECTOR_CACHE_V2.md)

---
© 2026 AstroTransit-GPU Team. Licensed under MIT.
