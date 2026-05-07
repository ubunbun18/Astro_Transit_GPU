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

### 7. `build-cache` — セクターキャッシュの構築
大量のバラバラな FITS ファイルを読み込み、前処理（NaN除去、正規化）を済ませた上で、GPU が一気飲みにできる一つの巨大なフラットバイナリファイル（NPZ）に集約します。

- **引数**:
  - `--fits-dir` (必須): MAST からダウンロードした FITS ファイル群が格納されているディレクトリ。
  - `--out-dir` (必須): キャッシュファイルの保存先。
  - `--workers` (Default: 8): FITS パースに使用する CPU 並列数。
- **使用例**:
  ```bash
  astrotransit-gpu build-cache --fits-dir data/tess_sector1 --out-dir data/sector1_cache
  ```

---

### 8. `screen-sector` — 超高速一括スクリーニング
`build-cache` で作成した集約データを用い、GPU リソースを 100% 活用してセクター内の全天体を一気に解析します。I/O 待ちがほぼゼロになるため、驚異的なスループットを発揮します。

- **引数**:
  - `--cache-dir` (必須): `build-cache` で作成したディレクトリ。
  - `--n-periods` (Default: 5000): 周期グリッドの密度。
  - `--precision` (Default: `float32`): 計算精度。
  - `--out` (Default: `screening_results.csv`): 結果の保存先。
- **使用例**:
  ```bash
  # 1.6万天体に対して10万周期の超精密探索を30分で実行
  astrotransit-gpu screen-sector --cache-dir data/sector1_cache --n-periods 100000
  ```

---

### 💡 Tips: 大量データのダウンロードについて
MAST 公式の `curl` スクリプトを利用して数万件のデータを一括取得するための補助スクリプトが同梱されています。

```bash
python scripts/bulk_download_sector.py --script path/to/mast_curl_script.sh --outdir data/tess_sector1 --threads 50
```
これにより、標準の `batch` コマンドよりも遥かに高速にデータセットを準備できます。

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
