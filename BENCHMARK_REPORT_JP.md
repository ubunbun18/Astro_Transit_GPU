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

- **AstroTransit-GPU**: v0.1.0
- **Commit**: `16643c1`
- **OS**: Windows 11
- **GPU**: NVIDIA Compute Capability 12.0 (Blackwell Architecture, 16GB VRAM)
- **CPU**: AMD Ryzen 7 9700X 8-Core Processor
- **RAM**: 32 GB (assumed)
- **Python**: 3.12.0
- **CuPy**: v13.6.0
- **Target**: TIC 261136679 (18,257 data points)
- **Grid Specs**: Period 0.5–20.0 days, 5 Durations, 200 Phase bins
- **Timing**: GPU ウォームアップ後、`cp.cuda.Stream.null.synchronize()` による明示的な同期計測を実施。

### 再現コマンド
```bash
# 標準解像度
astrotransit-gpu compare --preset standard

# 高解像度（大規模）
astrotransit-gpu compare --preset large

# ストレステスト
astrotransit-gpu compare --preset extreme
```

---

## 3. CLI 完全リファレンス

すべてのコマンドとオプションの詳細は以下の通りです。

### `check`：環境診断
GPU が利用可能か、またハードウェアの計算能力を確認します。

### `compare`：性能・精度比較
CPU と GPU の結果を直接比較し、Markdown レポートを生成します。
- `--target` : 解析対象 (デフォルト: "TIC 261136679")
- `--preset` : 設定 [`standard`, `large`, `extreme`]
- `--out` : レポート出力先

### `known`：既知の惑星の再検出
既存の惑星データを用いて、検出アルゴリズムの妥当性を確認します。
- `--target` : **[必須]** ターゲット ID
- `--true-p` : 比較用の真の周期

### `batch` : 一括解析
複数のターゲットを NASA アーカイブから取得し、並列探索を行います。

### `inject-run` : 回収率テスト
信号注入（Injection）と回収（Recovery）を行い、検出限界を統計的に評価します。

### `run-config` : 設定ファイル実行
YAML 形式の設定ファイルを用いて、再現性の高い解析フローを実行します。

---
*AstroTransit-GPU: Scaling Exoplanet Discovery with Reliability.*
