# 科学的検証（Validation）プロトコル

AstroTransit-GPU は、高速性だけでなく科学的な正確性を重視しています。本ドキュメントでは、本プロジェクトにおけるトランジット検出の妥当性評価の手順と基準について定義します。

## 1. 検証の目的
GPU カーネル（V39 等）の変更が、既知惑星の回収率（Completeness）や周期特定精度（Accuracy）にどのような影響を与えるかを定量的かつ再現可能に評価すること。

## 2. 公式判定基準 (Standard Criteria)
再現性を確保するため、以下の基準を `configs/validation_v39.yaml` に固定しています。

| パラメータ | 設定値 | 説明 |
| :--- | :--- | :--- |
| **SNR 閾値** | 7.1 | TESS 標準の有意性閾値。これ以上の Power を検出とみなす。 |
| **周期許容誤差 (p_tol)** | 0.02 (2%) | 検出周期 $P_{det}$ と真の周期 $P_{true}$ の相対誤差が 2% 以内。 |
| **T0 許容誤差 (t0_tol)** | 0.5 日 | トランジット時刻のズレの許容範囲。 |
| **T0 チェック (require_t0)** | True | 周期だけでなく時刻も一致していることを要求する。 |

## 3. 検証手順 (Reproduction Steps)

### 3.1 事前準備
検証には全天スクリーニングの結果 CSV（約 25MB）が必要です。
```bash
# セクター規模のスクリーニングを実行（未実行の場合）
python -m astrotransit_gpu screen-sector --blackwell --out outputs/bench_219331_v39.csv
```

### 3.2 検証スクリプトの実行
標準設定を用いて統計レポートを生成します。
```bash
# 旧: python scripts/reproduce_v39_validation.py
# 新: CLI サブコマンドから実行可能
astrotransit-gpu validate --results outputs/bench_219331_v39.csv
```

## 4. 指標の定義

### 4.1 完備性 (Completeness)
物理的に検出可能な（観測期間中に 2 回以上のトランジットが発生する）既知 TOI のうち、上記判定基準を満たして検出された割合。
$$Completeness = \frac{Recovered\ TOIs}{Detectable\ TOIs}$$

> [!IMPORTANT]
> **公式検証値と設定ファイルの使い分け**:
> 本プロジェクトでは、検証データの規模（入手範囲）に応じて 2 つの公式基準を定義しています。
>
> 1. **大規模サブセット基準 (Large-Scale Subset)**:
>    - **設定**: `configs/validation_v39_large_subset.yaml` (または `validation_v39.yaml`)
>    - **対象**: 入手済みの 219,331 天体（全天の一部のサブセット）
>    - **期待値**: **17.31%** (`require_t0: true`)
>    - **用途**: 入手できた広範囲なデータにおける公式なパイプライン性能。
>
> 2. **Sector 1 限定基準 (Sector 1 Focused)**:
>    - **設定**: `configs/validation_v39_sector1.yaml`
>    - **対象**: 約 16,000 天体（Sector 1 のみ）
>    - **期待値**: **38.75%** (`require_t0: false`)
>    - **用途**: 開発中のクイックな性能回帰テスト。

### 4.2 偽陽性率 (FPR / Unknown Signal Rate)
有意な信号（SNR > 7.1）のうち、既知の TOI または 食連星（EB）カタログに該当しない天体の割合。
※ 現時点では精査（Vetting）前の「候補天体」をすべて含んでいるため、高い値を示します。

## 5. カタログソース
検証の再現性を担保するため、以下のソースを使用します。
- **TOI カタログ**: NASA Exoplanet Archive (自動取得)
- **EB カタログ**: `data/catalogs/eb_latest.csv` (プロジェクト同梱)

オンラインアクセスが失敗した場合は、`data/catalogs/` 下のローカルキャッシュが自動的に使用されます。
