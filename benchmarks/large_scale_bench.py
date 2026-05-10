import pandas as pd
import numpy as np
import os
import logging
from astrotransit_gpu.validate.match import match_candidate
from astrotransit_gpu.data.exoplanet_archive import ExoplanetArchiveClient

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LargeScaleValidator:
    """21.9万件の大規模検出結果を検証するクラス"""

    def __init__(self, p_tol=0.02, power_threshold=7.1, t0_tol=0.5):
        self.p_tol = p_tol
        self.power_threshold = power_threshold
        self.t0_tol = t0_tol

    def validate_results(self, results_df, toi_df):
        """
        検出結果とTOIカタログを照合する。
        
        Args:
            results_df (pd.DataFrame): 検出結果 (tic_id, period, t0, power, ...)
            toi_df (pd.DataFrame): TOIカタログ (tid, pl_orbper, pl_tranmid, ...)
            
        Returns:
            dict: 統計サマリー
            pd.DataFrame: マッチング結果
            pd.DataFrame: 新候補天体
        """
        logger.info("Starting validation...")
        
        if results_df.empty:
            logger.warning("Results dataframe is empty.")
            return self._empty_results()

        # 1. 前処理: 型の統一と不要データの削除
        results_df['tic_id'] = results_df['tic_id'].astype(str)
        toi_df['tid'] = toi_df['tid'].astype(str)
        
        # 1.1 EBカタログのロード
        eb_df = pd.DataFrame()
        eb_path = "data/tess_eb_catalog.csv"
        if os.path.exists(eb_path):
            try:
                eb_df = pd.read_csv(eb_path)
                if 'tess_id' in eb_df.columns:
                    eb_df['tess_id'] = eb_df['tess_id'].astype(str)
                    logger.info(f"Loaded {len(eb_df)} EBs from catalog.")
                else:
                    logger.warning(f"Column 'tess_id' not found in EB catalog. Columns: {eb_df.columns.tolist()}")
            except Exception as e:
                logger.warning(f"Failed to parse EB catalog: {e}")
        
        # 2. サンプル内に存在するTOIの特定
        sample_tic_ids = set(results_df['tic_id'])
        tois_in_sample = toi_df[toi_df['tid'].isin(sample_tic_ids)].copy()
        
        # 2.1 物理的に検出可能な天体に絞り込む (Period < 13.7 days for at least 2 transits in S1)
        baseline = 27.4
        detectable_tois = tois_in_sample[tois_in_sample['pl_orbper'] < (baseline / 2.0)].copy()
        
        logger.info(f"Total targets in results: {len(results_df)}")
        logger.info(f"TOIs present in sample: {len(tois_in_sample)}")
        logger.info(f"Physically detectable TOIs (P < 13.7d): {len(detectable_tois)}")

        # 3. マッチング処理の高速化
        results_dict = results_df.set_index('tic_id').to_dict('index')
        matches = []
        
        for _, toi in detectable_tois.iterrows():
            tic_id = toi['tid']
            if tic_id not in results_dict:
                continue
            
            det = results_dict[tic_id]
            
            # T0の規格化 (BJD -> BTJD)
            t0_true = toi['pl_tranmid']
            if t0_true > 2450000:
                t0_true -= 2457000
                
            # match_candidate ロジック適用 (論文基準: p_tol=2%, t0_tol=0.5d)
            res = match_candidate(
                p_detected=det['period'],
                t0_detected=det['t0'],
                p_true=toi['pl_orbper'],
                t0_true=t0_true,
                p_tol=self.p_tol,
                t0_tol=self.t0_tol,
                require_t0=True
            )
            
            # ハーモニクス制限 (1, 2, 3, 1/2, 1/3 以外は無効化)
            allowed_harmonics = {"direct", "harmonic", "harmonic_3", "subharmonic", "subharmonic_3"}
            if res['match_type'] not in allowed_harmonics:
                res['is_match'] = False
                res['match_type'] = "none"

            res.update({
                'tic_id': tic_id,
                'det_power': det['power'],
                'det_period': det['period'],
                'true_period': toi['pl_orbper'],
                'toi_id': toi.get('toi', 'N/A')
            })
            matches.append(res)
            
        matches_df = pd.DataFrame(matches)
        
        # 4. 統計計算
        if not matches_df.empty:
            recovered_count = matches_df[matches_df['is_match']].shape[0]
            completeness = recovered_count / len(detectable_tois) if len(detectable_tois) > 0 else 0
        else:
            recovered_count = 0
            completeness = 0
            
        # 5. 新候補の抽出（カタログ未登録で高Powerなもの）
        toi_tic_ids = set(toi_df['tid'])
        eb_tic_ids = set(eb_df['tess_id']) if not eb_df.empty else set()
        
        significant_detections = results_df[results_df['power'] > self.power_threshold]
        
        # 既知天体（TOI or EB）の特定
        def is_known(row):
            tic_id = str(row['tic_id'])
            # TIC IDが一致するか、周期が一致するか
            if tic_id in toi_tic_ids: return "TOI"
            if tic_id in eb_tic_ids: return "EB"
            return "None"

        significant_detections = significant_detections.copy()
        significant_detections['known_type'] = significant_detections.apply(is_known, axis=1)
        
        new_candidates = significant_detections[significant_detections['known_type'] == "None"].copy()
        new_candidates = new_candidates.sort_values('power', ascending=False)
        
        # 統計計算 (FPR)
        total_sigs = len(significant_detections)
        fp_count = len(new_candidates)
        fpr = fp_count / total_sigs if total_sigs > 0 else 0
        
        summary = {
            'total_targets': len(results_df),
            'toi_in_sample': len(tois_in_sample),
            'detectable_toi': len(detectable_tois),
            'recovered_toi': recovered_count,
            'completeness': completeness,
            'new_candidates_count': len(new_candidates),
            'significant_sigs': total_sigs,
            'fpr': fpr,
            'eb_found': (significant_detections['known_type'] == "EB").sum()
        }
        
        # 6. 詳細分析 (Completeness Map & Period Accuracy)
        detail_stats = self._calculate_detailed_stats(matches_df, detectable_tois)
        
        # 6.1 物理パラメータ分析 (Tmag, Spectral Type)
        phys_stats = self._calculate_physical_stats(matches_df, detectable_tois)
        
        return summary, matches_df, new_candidates, detail_stats, phys_stats

    def _calculate_physical_stats(self, matches_df, detectable_tois):
        """Tmagや星のタイプ別の完備性を計算"""
        stats = {}
        if detectable_tois.empty: return stats
        
        matched_only = matches_df[matches_df['is_match']].copy()
        
        # 1. Tmag Bins
        tmag_bins = [6, 8, 10, 12, 14, 16]
        tmag_results = []
        # TOIテーブルからTmagを取得 (ティックIDで紐付け)
        for i in range(len(tmag_bins)-1):
            low, high = tmag_bins[i], tmag_bins[i+1]
            in_bin = detectable_tois[(detectable_tois['st_tmag'] >= low) & (detectable_tois['st_tmag'] < high)]
            if len(in_bin) > 0:
                recovered = matched_only[matched_only['tic_id'].isin(in_bin['tid'])]
                tmag_results.append({
                    'bin': f"{low}-{high}",
                    'total': len(in_bin),
                    'recovered': len(recovered),
                    'rate': len(recovered) / len(in_bin)
                })
        stats['tmag_completeness'] = tmag_results
        
        # 2. Spectral Type (by Teff)
        # M < 3700, K 3700-5200, G 5200-6000, F > 6000
        teff_bins = [
            ("M (<3700K)", 0, 3700),
            ("K (3700-5200K)", 3700, 5200),
            ("G (5200-6000K)", 5200, 6000),
            ("F (>6000K)", 6000, 10000)
        ]
        type_results = []
        for label, low, high in teff_bins:
            in_bin = detectable_tois[(detectable_tois['st_teff'] >= low) & (detectable_tois['st_teff'] < high)]
            if len(in_bin) > 0:
                recovered = matched_only[matched_only['tic_id'].isin(in_bin['tid'])]
                type_results.append({
                    'type': label,
                    'total': len(in_bin),
                    'recovered': len(recovered),
                    'rate': len(recovered) / len(in_bin)
                })
        stats['type_completeness'] = type_results
        
        return stats

    def _calculate_detailed_stats(self, matches_df, detectable_tois):
        """周期・深さごとの完備性と周期精度を計算"""
        stats = {}
        
        if matches_df.empty or detectable_tois.empty:
            return stats

        # 1. Period Accuracy (Relative Error %)
        matched_only = matches_df[matches_df['is_match']].copy()
        if not matched_only.empty:
            rel_errors = matched_only['p_diff'] * 100
            stats['period_accuracy'] = {
                'median': np.median(rel_errors),
                'p95': np.percentile(rel_errors, 95),
                'min': np.min(rel_errors),
                'max': np.max(rel_errors)
            }

        # 2. Completeness by Period Bins
        p_bins = [0, 2, 5, 10, 13.7]
        p_completeness = []
        for i in range(len(p_bins)-1):
            low, high = p_bins[i], p_bins[i+1]
            in_bin = detectable_tois[(detectable_tois['pl_orbper'] >= low) & (detectable_tois['pl_orbper'] < high)]
            if len(in_bin) > 0:
                # matches_dfの中でこのbinに属するTOI IDを特定
                matched_in_bin = matched_only[matched_only['true_period'].between(low, high)]
                p_completeness.append({
                    'bin': f"{low}-{high}d",
                    'total': len(in_bin),
                    'recovered': len(matched_in_bin),
                    'rate': len(matched_in_bin) / len(in_bin)
                })
        stats['p_completeness'] = p_completeness
        
        return stats

    def _empty_results(self):
        summary = {
            'total_targets': 0, 'toi_in_sample': 0, 'detectable_toi': 0, 
            'recovered_toi': 0, 'completeness': 0.0, 'new_candidates_count': 0,
            'significant_sigs': 0, 'fpr': 0.0, 'eb_found': 0
        }
        return summary, pd.DataFrame(), pd.DataFrame(), {}, {}

def main():
    # パスの設定
    results_path = "outputs/bench_219331_v39.csv"
    report_path = "reports/MASSIVE_VALIDATION_REPORT_JP.md"
    
    if not os.path.exists(results_path):
        logger.error(f"Results file not found: {results_path}")
        return

    # 1. 検出結果のロード
    logger.info(f"Loading results from {results_path}...")
    results_df = pd.read_csv(results_path)
    
    # 2. TOIカタログの取得
    try:
        toi_df = ExoplanetArchiveClient.get_toi_table()
    except Exception as e:
        logger.error(f"Failed to fetch TOI table: {e}")
        return

    # 3. 検証の実行
    validator = LargeScaleValidator(p_tol=0.02, power_threshold=7.1, t0_tol=0.5)
    summary, matches_df, new_candidates, detail_stats, phys_stats = validator.validate_results(results_df, toi_df)
    
    # 4. レポートの生成
    logger.info("Generating report...")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# AstroTransit-GPU 大規模検証レポート (v1.3.0 - V39 Standard)\n\n")
        f.write(f"解析日時: {pd.Timestamp.now()}\n")
        f.write(f"使用カーネル: V39 Apex Predator (Weight-aware/SNR-Normalized)\n")
        f.write(f"設定: p_tol=0.10, power_threshold=7.1 (TESS Standard)\n\n")
        
        f.write("## 1. 統計サマリー\n")
        f.write(f"- **総解析天体数**: {summary['total_targets']:,}\n")
        f.write(f"- **サンプル内の既知TOI数**: {summary['toi_in_sample']:,}\n")
        f.write(f"- **物理的に検出可能なTOI数 (P < 13.7d)**: {summary['detectable_toi']:,}\n")
        f.write(f"- **再検出に成功したTOI数**: {summary['recovered_toi']:,}\n")
        f.write(f"- **完備性 (Completeness / Detectable Recovery)**: **{summary['completeness']:.2%}**\n")
        f.write(f"- **有意な信号数 (SNR > 7.1)**: {summary['significant_sigs']:,}\n")
        f.write(f"- **既知の食連星(EB)数**: {summary['eb_found']:,}\n")
        f.write(f"- **偽陽性率 (FPR / Unknown Signal Rate)**: **{summary['fpr']:.2%}**\n")
        f.write(f"- **新候補天体数 (Potential ND)**: {summary['new_candidates_count']:,}\n\n")
        
        f.write("## 2. 詳細分析\n")
        
        # 2.1 Period Recovery Accuracy
        if 'period_accuracy' in detail_stats:
            pa = detail_stats['period_accuracy']
            f.write("### 2.1 周期検出精度 (相対誤差 %)\n")
            f.write(f"- **Median Error**: {pa['median']:.4f}%\n")
            f.write(f"- **95th Percentile**: {pa['p95']:.4f}%\n")
            f.write(f"- **Range**: {pa['min']:.4f}% - {pa['max']:.4f}%\n\n")
            
        # 2.2 Completeness Map (Period)
        if 'p_completeness' in detail_stats:
            f.write("### 2.2 周期別完備性 (Completeness Map)\n")
            f.write("| Period Bin | Total TOIs | Recovered | Rate |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for entry in detail_stats['p_completeness']:
                f.write(f"| {entry['bin']} | {entry['total']} | {entry['recovered']} | {entry['rate']:.2%} |\n")
            f.write("\n")

        # 2.3 Physical Parameter Analysis
        f.write("### 2.3 物理パラメータ別完備性\n")
        if 'tmag_completeness' in phys_stats:
            f.write("#### 等級別 (Tmag)\n")
            f.write("| Tmag Bin | Total | Recovered | Rate |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for entry in phys_stats['tmag_completeness']:
                f.write(f"| {entry['bin']} | {entry['total']} | {entry['recovered']} | {entry['rate']:.2%} |\n")
            f.write("\n")
            
        if 'type_completeness' in phys_stats:
            f.write("#### 星タイプ別 (Spectral Type)\n")
            f.write("| Type (Teff) | Total | Recovered | Rate |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for entry in phys_stats['type_completeness']:
                f.write(f"| {entry['type']} | {entry['total']} | {entry['recovered']} | {entry['rate']:.2%} |\n")
            f.write("\n")

        f.write("## 3. マッチングの内訳\n")
        if not matches_df.empty:
            type_counts = matches_df[matches_df['is_match']]['match_type'].value_counts()
            for m_type, count in type_counts.items():
                f.write(f"- {m_type}: {count} 件\n")
        
        f.write("\n## 4. 上位の新候補 (Top 20)\n")
        if not new_candidates.empty:
            f.write("| TIC ID | Power | Period (days) | Depth |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for _, row in new_candidates.head(20).iterrows():
                f.write(f"| {row['tic_id']} | {row['power']:.2f} | {row['period']:.4f} | {row['depth']:.6f} |\n")
        else:
            f.write("該当なし\n")

    logger.info(f"Validation complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
