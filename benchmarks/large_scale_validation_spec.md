# CbC Specification: Large-Scale Transit Validation

## 1. インターフェース定義

### クラス名: `LargeScaleValidator`

#### メソッド: `validate_results`

**入力:**
- `results_path` (str): 検出結果CSVのパス (`outputs/bench_219331.csv`)
- `toi_catalog` (pd.DataFrame): NASA Exoplanet Archiveから取得したTOIカタログデータフレーム
- `p_tol` (float): 周期の相対許容誤差 (デフォルト 0.01)
- `power_threshold` (float): 有効な検出とみなすPowerの閾値 (デフォルト 10.0)

**出力:**
- `dict`:
    - `summary`:
        - `total_targets`: 解析対象の総天体数
        - `toi_in_sample`: サンプルに含まれていたTOIの数
        - `recovered_toi`: マッチングに成功したTOIの数
        - `recovery_rate`: 回収率 (recovered / toi_in_sample)
        - `new_candidates_count`: カタログ未登録で高Powerな天体数
    - `matches_df`: 各TOIに対するマッチング詳細 (DataFrame)
    - `new_candidates_df`: カタログ未登録の高Power天体リスト (DataFrame)

---

## 2. 詳細ロジック（擬似コード）

```python
def validate_results(results_df, toi_df, p_tol=0.01, power_threshold=10.0):
    # 1. 検出結果のフィルタリング
    # Powerが閾値以下のものはノイズとして扱う
    significant_detections = results_df[results_df['power'] > power_threshold]
    
    # 2. サンプル内に存在するTOIの特定
    sample_tic_ids = set(results_df['tic_id'].astype(str))
    tois_in_sample = toi_df[toi_df['tic_id'].astype(str).isin(sample_tic_ids)]
    
    matches = []
    
    # 3. 各TOIに対してマッチング試行
    # TIC IDごとの検索を高速化するため辞書化
    results_dict = results_df.set_index('tic_id').to_dict('index')
    
    for _, toi in tois_in_sample.iterrows():
        tic_id = str(toi['tic_id'])
        if tic_id not in results_dict:
            continue
            
        det = results_dict[tic_id]
        
        # 周期の比較
        p_det = det['period']
        p_true = toi['period']
        t0_det = det['t0']
        t0_true = toi['t0']
        
        # 既存のマッチングロジック (match_candidate) を使用
        match_info = match_candidate(p_det, t0_det, p_true, t0_true, p_tol=p_tol)
        match_info['tic_id'] = tic_id
        match_info['det_power'] = det['power']
        match_info['true_period'] = p_true
        match_info['det_period'] = p_det
        matches.append(match_info)
        
    # 4. 新候補の抽出（カタログにない高Power天体）
    toi_tic_ids = set(toi_df['tic_id'].astype(str))
    new_candidates = significant_detections[~significant_detections['tic_id'].astype(str).isin(toi_tic_ids)]
    new_candidates = new_candidates.sort_values('power', ascending=False)
    
    return summary_stats, pd.DataFrame(matches), new_candidates
```

---

## 3. 不変条件 (Invariants)

1.  **非負性**: 周期、Power、天体数は常に 0 以上である。
2.  **回収率の範囲**: 回収率は 0.0 以上 1.0 以下の範囲に収まる。
3.  **データ型**: `tic_id` の照合時は、型（int vs str）の不一致を防ぐため明示的に変換する。

---

## 4. テストケース

### 4.1 正常系
- **Case 1**: サンプル内の既知TOIが「direct match」として正しく識別される。
- **Case 2**: 周期が真の2倍で検出された場合、「harmonic match」として識別される。
- **Case 3**: カタログにない高Power天体が `new_candidates_df` に含まれる。

### 4.2 境界値テスト
- **Case 4**: `power` がちょうど `power_threshold` の場合。
- **Case 5**: 周期誤差がちょうど `p_tol` (e.g. 1%) の場合。

### 4.3 異常系・契約違反テスト
- **Case 6**: `results_df` が空の場合（エラーにならず、回収率 0 となるべき）。
- **Case 7**: 周期が 0 または 負数のデータが含まれる場合（事前チェックで弾くべき）。

---

## 5. 期待される成果物
- `benchmarks/large_scale_bench.py`: 本仕様に基づく実装。
- `reports/MASSIVE_VALIDATION_REPORT_JP.md`: 21.9万件の検証結果レポート。
