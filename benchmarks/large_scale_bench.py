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

    def __init__(self, p_tol=0.01, power_threshold=10.0):
        self.p_tol = p_tol
        self.power_threshold = power_threshold

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
        
        # 2. サンプル内に存在するTOIの特定
        sample_tic_ids = set(results_df['tic_id'])
        tois_in_sample = toi_df[toi_df['tid'].isin(sample_tic_ids)].copy()
        
        logger.info(f"Total targets in results: {len(results_df)}")
        logger.info(f"TOIs present in sample: {len(tois_in_sample)}")

        # 3. マッチング処理の高速化
        results_dict = results_df.set_index('tic_id').to_dict('index')
        matches = []
        
        for _, toi in tois_in_sample.iterrows():
            tic_id = toi['tid']
            if tic_id not in results_dict:
                continue
            
            det = results_dict[tic_id]
            
            # T0の規格化 (BJD -> BTJD)
            t0_true = toi['pl_tranmid']
            if t0_true > 2450000:
                t0_true -= 2457000
                
            # match_candidate ロジック適用 (T0マッチングを緩和)
            res = match_candidate(
                p_detected=det['period'],
                t0_detected=det['t0'],
                p_true=toi['pl_orbper'],
                t0_true=t0_true,
                p_tol=self.p_tol,
                require_t0=False  # 大規模検証ではまず周期の一致を優先
            )
            
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
            recovery_rate = recovered_count / len(tois_in_sample) if len(tois_in_sample) > 0 else 0
        else:
            recovered_count = 0
            recovery_rate = 0
            
        # 5. 新候補の抽出（カタログ未登録で高Powerなもの）
        toi_tic_ids = set(toi_df['tid'])
        significant_detections = results_df[results_df['power'] > self.power_threshold]
        new_candidates = significant_detections[~significant_detections['tic_id'].isin(toi_tic_ids)].copy()
        new_candidates = new_candidates.sort_values('power', ascending=False)
        
        summary = {
            'total_targets': len(results_df),
            'toi_in_sample': len(tois_in_sample),
            'recovered_toi': recovered_count,
            'recovery_rate': recovery_rate,
            'new_candidates_count': len(new_candidates)
        }
        
        return summary, matches_df, new_candidates

    def _empty_results(self):
        return {
            'total_targets': 0, 'toi_in_sample': 0, 'recovered_toi': 0, 
            'recovery_rate': 0, 'new_candidates_count': 0
        }, pd.DataFrame(), pd.DataFrame()

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
    # TESS標準のSNRしきい値(7.1)を採用し、周期誤差を10%まで緩和（倍音やFFIの精度を考慮）
    validator = LargeScaleValidator(p_tol=0.10, power_threshold=7.1)
    summary, matches_df, new_candidates = validator.validate_results(results_df, toi_df)
    
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
        f.write(f"- **再検出に成功したTOI数**: {summary['recovered_toi']:,}\n")
        f.write(f"- **回収率 (Recovery Rate)**: {summary['recovery_rate']:.2%}\n")
        f.write(f"- **高SNRの新候補数 (SNR > 7.1)**: {summary['new_candidates_count']:,}\n\n")
        
        f.write("## 2. マッチングの内訳\n")
        if not matches_df.empty:
            type_counts = matches_df[matches_df['is_match']]['match_type'].value_counts()
            for m_type, count in type_counts.items():
                f.write(f"- {m_type}: {count} 件\n")
        
        f.write("\n## 3. 上位の新候補 (Top 20)\n")
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
