# AstroTransit-GPU CLI 完全リファレンス (v1.0.0)

AstroTransit-GPU のコマンドラインインターフェースは、研究者が再現可能な解析を迅速に行えるよう設計されています。

---

## 🚀 共通仕様
- **単位**: 周期は「日 (days)」、深さは「相対光度差 (relative depth)」を基本とします。
- **ターゲット指定**: TESS の場合は `TIC 123456`、Kepler の場合は `KIC 123456` 形式をサポートします。

---

## 🛠️ コマンド詳細

### 1. `check` — 環境診断
現在のシステムで CUDA が正しく認識されているか、GPU のスペックを確認します。

- **使用例**:
  ```bash
  astrotransit-gpu check
  ```
- **出力内容**:
  - CUDA の利用可否
  - GPU デバイス名
  - Compute Capability (8.6 以上推奨)

---

### 2. `search` — 単一ターゲット探索
特定の TIC ID に対して高速な BLS 探索を直接実行します。

- **引数**:
  - `--target` (必須): ターゲットの TIC ID。
  - `--n-periods` (Default: 5000): 探索する周期の数。
  - `--precision` (Default: `float32`): 計算精度。`float32` または `float64`。
  - `--out`: 結果（JSON形式）の保存先パス。
- **使用例**:
  ```bash
  astrotransit-gpu search --target "TIC 261136679" --n-periods 10000 --precision float32
  ```

---

### 3. `compare` — 性能・精度検証
CPU (Astropy) と GPU の結果を直接比較し、計算の正確性と加速倍率を検証します。

- **引数**:
  - `--target` (Default: "TIC 261136679"): 検証に使用する天体。
  - `--preset`: 探索規模のプリセット。
    - `standard`: 5,000 周期 (デフォルト)
    - `large`: 100,000 周期
    - `extreme`: 1,000,000 周期
  - `--n-runs` (Default: 5): 平均実行時間を計算するための試行回数。
  - `--out` (Default: `comparison_report.md`): パリティレポートの保存先。
- **使用例**:
  ```bash
  astrotransit-gpu compare --preset large --n-runs 3
  ```

---

### 4. `inject` — 回収率テスト (Injection/Recovery)
人工的な信号を注入し、現在のデータ品質でどの程度の惑星が検出可能か（Recovery Map）を評価します。

- **引数**:
  - `--target`: ベースとなる光度曲線データ。
  - `--periods` (Default: "2.0,5.0,10.0"): 注入する周期（カンマ区切り）。
  - `--depths` (Default: "0.001,0.005,0.01"): 注入するトランジットの深さ。
  - `--n-trials` (Default: 5): 各セルあたりの試行回数。
  - `--out` (Default: `injection_report.md`): 最終的な回収率マップの出力先。
- **使用例**:
  ```bash
  astrotransit-gpu inject --periods "1.5,3.0,5.0,10.0" --depths "0.0005,0.001,0.005" --n-trials 10
  ```

---

### 5. `benchmark` — 再現可能ベンチマーク実行
YAML 設定ファイルに基づき、論文や報告書に使用できる詳細なレポートと図を自動生成します。

- **引数**:
  - `--config` (必須): 解析条件を記述した YAML ファイルのパス。
  - `--outdir` (Default: `reports`): レポート、画像（周期図、折り畳み光度曲線）、JSON データの保存ディレクトリ。
- **YAML 設定例**:
  ```yaml
  benchmark_id: "TESS_Standard_Run"
  target: "TIC 261136679"
  period_min: 0.5
  period_max: 20.0
  n_periods: 5000
  timed_runs: 5
  durations: [0.01, 0.05, 0.1]
  ```
### `benchmark`
再現可能な性能レポートを生成します。

```bash
astrotransit-gpu benchmark --config config.yaml [--outdir reports] [--gpu-only]
```

- `--config`: YAML 設定ファイルのパス。
- `--outdir`: レポート出力先ディレクトリ（デフォルト: `reports`）。
- `--gpu-only`: CPU (Astropy) での比較計測をスキップします。超大規模探索時に推奨。

---

### 6. `batch` — 一括並列解析 (ベータ)
### `batch`
多数の天体を一括解析します。非同期 I/O により通信待ちを最小化します。

```bash
astrotransit-gpu batch --targets targets.csv [--out results.csv] [--workers 4] [--resume]
```

- `--targets`: `tic_id` カラムを含む CSV ファイル。
- `--out`: 結果の保存先 CSV（デフォルト: `batch_results.csv`）。
- `--workers`: ダウンロードと前処理を行う並列スレッド数（デフォルト: 4）。
- `--resume`: 保存先 CSV を確認し、すでに `ok` ステータスの天体はスキップします。
- **堅牢性**: 破損した FITS ファイルを自動検知し、キャッシュを削除して再試行する機能を内蔵しています。
へのパス。
- **使用例**:
  ```bash
  astrotransit-gpu batch --targets candidate_list.csv
  ```

---

## 💡 Tips & トラブルシューティング

### 1. メモリ不足 (OutOfMemory)
`n_bins`（ビン数）や `N_TILE`（タイルサイズ）を大きくしすぎると、GPU の共有メモリ上限（48KB等）に達する場合があります。
- **対策**: `n_bins` を 500 以下に抑えるか、`--precision float32` を使用してください。

### 2. 精度と速度のトレードオフ
- 一般的なスクリーニングには `float32` で十分です。
- 100日以上の長い観測データで、非常に狭いトランジットを探索する場合は `float64` の使用を検討してください。

### 3. 出力の活用
`benchmark` コマンドで生成される `benchmark.json` は、独自に解析プログラムを作成する際の入力データとして最適です。
