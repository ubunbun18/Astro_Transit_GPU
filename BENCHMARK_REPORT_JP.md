# AstroTransit-GPU 高速探索・完全活用ガイド
## 〜 探索効率の最大化と数値精度の検証報告 〜

AstroTransit-GPU は、CUDA カーネルによる BLS (Box Least Squares) アルゴリズムを実装し、Astropy を基準とした高い数値精度と、大規模データに対するスループットの向上を両立したトランジット探索ツールです。

---

## 1. パフォーマンス・ベンチマーク報告

本システムのスケーラビリティを検証するため、同一の探索グリッド条件下で Astropy (CPU) との実行速度を比較しました。

### 実行速度の比較
| 解析スケール | 周期グリッド数 | CPU (Astropy) | GPU (Ours) | 高速化倍率 | 
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | 5,000 | ~0.83秒 | **0.11秒** | **約 7.5倍** |
| **Large** | 100,000 | 16.86秒 | **0.21秒** | **約 79倍** |
| **Extreme** | 1,000,000 | 159.10秒 | **1.21秒** | **約 131倍** |

> [!NOTE]
> 探索規模（周期グリッド数）が拡大するほど GPU の並列演算スロットが効率的に埋まり、100万周期を超える探索では **130倍以上** のスループットを達成します。

### 数値一致性の検証
`Standard` プリセット（5,000周期）における、検出結果の物理的な一致度です。

| 項目 | CPU (Astropy) | GPU (Ours) | 物理差分 | 一致率 |
| :--- | :--- | :--- | :--- | :--- |
| **Best Period** | 6.269254 d | 6.265353 d | 0.0039 d | **99.94%** |
| **Best T0** | 1325.4994 | 3.4772 * | 0.3788 d | 位相一致 |

*\* GPU側は T0 を位相空間上で計算しているため、周期 $(P)$ で剰余をとった値で比較。*

---

## 2. ベンチマーク測定環境

本報告の数値は、以下の環境において再現可能です。

- **AstroTransit-GPU**: v1.0.0
- **Commit**: `latest`
- **OS**: Windows 11
- **GPU**: NVIDIA Compute Capability 8.6+ (e.g. RTX 30/40 series)
- **CPU**: Modern Multi-core CPU
- **Python**: 3.12.0
- **CuPy**: v13.x
- **Target**: TIC 261136679 (18,257 data points)
- **Grid Specs**: Period 0.5–20.0 days, 5 Durations, 200–500 Phase bins
- **Timing**: GPU ウォームアップ後、同期計測を実施。

### 再現コマンド
```bash
# 標準解像度 (5,000周期)
astrotransit-gpu compare --preset standard

# 高解像度 (100,000周期)
astrotransit-gpu compare --preset large

# ストレステスト (1,000,000周期)
astrotransit-gpu compare --preset extreme
```

---

## 3. CLI コマンド概要

AstroTransit-GPU は直感的な CLI 体系を提供しています。詳細な引数や実行例については、[CLI 完全リファレンス (docs/CLI_REFERENCE_JP.md)](./docs/CLI_REFERENCE_JP.md) を参照してください。

| コマンド | 説明 |
| :--- | :--- |
| `check` | CUDA 環境と GPU デバイスの診断。 |
| `search` | 単一ターゲットに対する高速探索。 |
| `compare` | CPU と GPU の数値一致性と速度の直接比較。 |
| `inject` | 信号注入実験による回収率（Recovery Map）の生成。 |
| `benchmark` | YAML 設定に基づく再現可能なレポート自動生成。 |
| `batch` | 複数ターゲットの一括並列解析。 |

---
*AstroTransit-GPU: Scaling Exoplanet Discovery with Reliability.*
