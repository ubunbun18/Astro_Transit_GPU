# AstroTransit-GPU: Candidate Vetting Pipeline (V39)

AstroTransit-GPU の Vetting パイプラインは、GPU による超高速スクリーニングで得られた大量の候補から、真に有望な天体を短時間で精査（Triage）するための 3 段階のインテリジェント・ワークフローです。

## 1. ワークフロー概要

パイプラインは以下の 3 つのコマンドで構成されます。

### Step 1: Sector Screening
セクター全体を高速スキャンし、各天体の最有力候補を抽出します。
```bash
python -m astrotransit_gpu.cli screen-sector --cache-dir data/s1_cache --out s1_raw.csv
```

### Step 2: Candidate Refinement
独自の抽出ルール（SNR、既知天体、惑星らしさ等）に基づき、有望天体を詳細に再探索します。
```bash
python -m astrotransit_gpu.cli refine --results s1_raw.csv --cache-dir data/s1_cache --out s1_refined.csv --config configs/vetting_v1.yaml
```

### Step 3: Vetting & Dashboard Generation
倍数周期の整理、スコアリング、プロット生成、および Triage ダッシュボードの構築を行います。
```bash
python -m astrotransit_gpu.cli vet --results s1_refined.csv --cache-dir data/s1_cache --out reports/s1_vet --config configs/vetting_v1.yaml
```

## 2. 成果物構成

出力ディレクトリには以下のファイルが生成され、解析の再現性が完全に担保されます。

- `index.html`: **Triage Dashboard**（推奨：ブラウザで開き、候補を仕分ける）
- `summary.json`: 実行統計とメタデータ（機械読み取り用）
- `candidates_ranked.csv`: 全データを含むランキングリスト
- `plots/`: 高精度な folded light curve 画像（Raw + Binned 表示）

## 3. Triage ダッシュボードの特徴

ダッシュボード (`index.html`) は、研究者が「見るべきものを減らす」ために設計されています。

- **Summary Cards**: 全体候補数、未登録天体、ハイスコア候補を一目で把握。
- **Status Badges**: `TOI`（既知惑星）、`EB`（食連星）、`Unknown`（未登録）を色分け。
- **Harmonic Flag (H)**: 倍数周期による重複候補を自動マーク。
- **Automated Notes**: スコアの根拠（High Priority, Short Transit 等）を自動表示。
- **Interactive Review**: 404 エラーを回避する堅牢なプレビュー機能。オフラインでも動作。

## 4. スコアリングと再探索ルール (`vetting_v1.yaml`)

再探索（Refinement）では、以下の 6 つのルールでターゲットを救済・抽出します。
1. **SNR 閾値**: 絶対的な強度による抽出。
2. **Top-N**: 相対的な順位による救済。
3. **カタログ照合**: 既知の TOI/EB を確実に確認。
4. **Artifact/EB 疑い**: 非常に深い減光や長すぎる継続時間の抽出。
5. **惑星らしさ**: 低 SNR でも「浅く鋭い」トランジットを持つ天体の救済。
6. **ランダムサンプル**: 統計的バイアスの排除。

---

## 技術的な再現性
生成される `summary.json` には、使用された Kernel バージョン（V39 Apex Predator）、Config パス、入力ファイルが記録されます。
```json
{
    "kernel_version": "V39 Apex Predator",
    "config": "configs/vetting_v1.yaml",
    "total_targets": 3560,
    "high_score_unknowns": 1956
}
```
