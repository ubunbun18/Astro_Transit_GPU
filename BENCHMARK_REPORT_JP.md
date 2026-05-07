# AstroTransit-GPU 高速探索・完全活用ガイド
## 〜 100倍速の宇宙探査：ベンチマーク ＆ CLIリファレンス 〜

AstroTransit-GPU は、CUDA カーネルを極限まで最適化することで、従来の CPU ベースの解析を圧倒するスピードで惑星探査を可能にします。このドキュメントでは、その驚異的な性能データと、全機能を使いこなすための CLI リファレンスを網羅しています。

---

## 1. パフォーマンス・ベンチマーク（実測値）

NVIDIA GPU（Blackwell / RTX シリーズ）環境での、CPU（Astropy BLS）との直接比較データです。

| 探索スケール | 周期グリッド数 | CPU (Astropy) | GPU (Ours) | 高速化倍率 | 推奨用途 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | 5,000 | ~0.85秒 | **0.04秒** | **約 21倍** | クイックチェック |
| **Large** | 100,000 | 16.86秒 | **0.21秒** | **約 79倍** | 標準的な個別解析 |
| **Extreme** | 1,000,000 | 159.10秒 | **1.21秒** | **約 131倍** | 超高密度・全天探索 |

> [!IMPORTANT]
> 探索規模が大きくなるほど、GPU の並列演算ユニットが効率的に活用され、**100倍以上のパフォーマンス・アドバンテージ**が得られます。

---

## 2. CLI 完全リファレンス

すべてのコマンドとオプションの詳細です。

### 1️⃣ `check`：環境の健全性確認
GPU が正しく認識されているか、計算能力（Compute Capability）が十分かを確認します。
- **使用例**: `astrotransit-gpu check`
- **引数**: なし

### 2️⃣ `compare`：CPU vs GPU 性能比較
現在の環境でどれだけの速度向上が得られるか、また数値的な一致度を検証します。
- **使用例**: `astrotransit-gpu compare --preset large --out my_bench.md`
- **オプション**:
    - `--target` : 解析対象の TIC ID (デフォルト: "TIC 261136679")
    - `--n-periods` : 手動で指定する周期グリッド数 (デフォルト: 5000)
    - `--preset` : 高速設定 [`standard` (5k), `large` (100k), `extreme` (1M)]
    - `--out` : 結果レポートの保存先 (デフォルト: "comparison_report.md")

### 3️⃣ `known`：既知の惑星の再検出
カタログに載っている既知の惑星を検出し、その精度を検証します。
- **使用例**: `astrotransit-gpu known --target "TIC 261136679" --true-p 6.26 --out report.md`
- **オプション**:
    - `--target` : **[必須]** ターゲット ID
    - `--mission` : ミッション選択 [`tess`, `kepler`] (デフォルト: "tess")
    - `--true-p` : 既知の真の周期 (比較用)
    - `--true-t0` : 既知の真の T0 (比較用)
    - `--out` : 出力ファイル名 (デフォルト: "report.md")

### 4️⃣ `batch`：一括解析（ハイスピード・サーチ）
NASA Exoplanet Archive からターゲットを自動抽出し、連続的に解析を行います。
- **使用例**: `astrotransit-gpu batch --n-targets 20 --min-depth 1500`
- **オプション**:
    - `--n-targets` : 取得・解析するターゲット数 (デフォルト: 10)
    - `--min-depth` : カタログ上の最低トランジット深度 [ppm] (デフォルト: 500.0)
    - `--out` : バッチレポートの保存先 (デフォルト: "batch_report.md")

### 5️⃣ `inject-run`：科学的妥当性の検証
光度曲線に人工的なトランジット信号を注入し、何パーセント回収できるかをテストします。
- **使用例**: `astrotransit-gpu inject-run --target "TIC 261136679" --periods "2,5,10" --n-trials 10`
- **オプション**:
    - `--target` : **[必須]** ベースとなる光度曲線の ID
    - `--periods` : 注入する周期（カンマ区切り） (デフォルト: "2.0,5.0,10.0")
    - `--depths` : 注入する深度（カンマ区切り） (デフォルト: "0.001,0.003,0.01")
    - `--n-trials` : 各セルごとの試行回数 (デフォルト: 3)
    - `--out` : 解析レポートの保存先 (デフォルト: "injection_recovery_report.md")

### 6️⃣ `run-config`：構成ファイルによる実行
YAML 形式の設定ファイルを使用して、複雑な探索条件を一括実行します。
- **使用例**: `astrotransit-gpu run-config configs/search_param.yaml`
- **引数**:
    - `config` : **[必須]** YAML 設定ファイルへのパス

---

## 3. ヒント ＆ トラブルシューティング

- **GPU メモリ不足**: `extreme` プリセット等で 1000万周期を超えるような探索を行う場合、GPU メモリ（VRAM）の消費量が増大します。エラーが出る場合は `--n-periods` を下げて調整してください。
- **高速化のコツ**: バッチ探索（`batch`）を使用する際は、ネットワーク速度（Lightkurve のダウンロード速度）がボトルネックになることがあります。安定した回線環境での実行を推奨します。

---
*Scaling Exoplanet Discovery with AstroTransit-GPU*
