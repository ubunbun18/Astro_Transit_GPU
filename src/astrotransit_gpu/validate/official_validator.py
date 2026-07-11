import os
import yaml
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
from ..data.exoplanet_archive import ExoplanetArchiveClient
from ..validate.match import match_candidate

logger = logging.getLogger(__name__)

# Completeness evaluation constants.
# A TESS sector baseline is ~27.4 days. To observe at least MIN_TRANSITS
# transits within the baseline, the period must satisfy
# (MIN_TRANSITS - 1) * P < SECTOR_BASELINE_DAYS, i.e. P < baseline / (MIN_TRANSITS - 1).
# With MIN_TRANSITS = 2 this gives P < 27.4 days (two transits: at t=0 and at t=P).
SECTOR_BASELINE_DAYS = 27.4
MIN_TRANSITS_FOR_COMPLETENESS = 2
MAX_DETECTABLE_PERIOD_DAYS = SECTOR_BASELINE_DAYS / (MIN_TRANSITS_FOR_COMPLETENESS - 1)

class OfficialValidator:
    """
    V39以降の科学検証を標準手順化するための公式検証クラス。
    YAML設定に基づき、再現可能な統計レポートを生成する。
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        v_conf = self.config.get('validation', {})
        self.power_threshold = v_conf.get('power_threshold', 7.1)
        self.p_tol = v_conf.get('p_tol', 0.01)
        self.t0_tol = v_conf.get('t0_tol', 0.5)
        self.require_t0 = v_conf.get('require_t0', False)
        
        cat_conf = self.config.get('catalogs', {})
        self.toi_path = cat_conf.get('toi_path', 'data/catalogs/toi_latest.csv')
        self.eb_path = cat_conf.get('eb_path', 'data/catalogs/eb_latest.csv')

    def load_catalogs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        TOIおよびEBカタログをロードする。オンライン取得に失敗した場合はローカルファイルを使用する。
        """
        # 1. TOI カタログ
        toi_df = None
        try:
            logger.info("Fetching TOI table from NASA Exoplanet Archive...")
            toi_df = ExoplanetArchiveClient.get_toi_table()
            # 成功したら将来のために保存しておく
            os.makedirs(os.path.dirname(self.toi_path), exist_ok=True)
            toi_df.to_csv(self.toi_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to fetch online TOI table: {e}. Falling back to local file: {self.toi_path}")
            if os.path.exists(self.toi_path):
                toi_df = pd.read_csv(self.toi_path)
            else:
                raise RuntimeError(f"TOI catalog not found online or at {self.toi_path}")

        # 2. EB カタログ
        eb_df = pd.DataFrame()
        if os.path.exists(self.eb_path):
            eb_df = pd.read_csv(self.eb_path)
            logger.info(f"Loaded {len(eb_df)} EBs from local catalog.")
        else:
            logger.warning(f"EB catalog not found at {self.eb_path}. Skipping EB exclusion.")
            
        return toi_df, eb_df

    def run_validation(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        大規模探索結果のバリデーションを実行し、統計情報を返す。
        """
        logger.info("Starting official validation pipeline...")
        toi_df, eb_df = self.load_catalogs()
        
        # TIC IDの型を統一
        results_df['tic_id'] = results_df['tic_id'].astype(str)
        toi_df['tid'] = toi_df['tid'].astype(str)
        if not eb_df.empty and 'tess_id' in eb_df.columns:
            eb_df['tess_id'] = eb_df['tess_id'].astype(str)
            eb_ids = set(eb_df['tess_id'])
        else:
            eb_ids = set()

        # 1. 有意な信号 (Significant Detections)
        significant_sigs = results_df[results_df['power'] > self.power_threshold].copy()
        
        # 2. サンプル内TOIの特定
        sample_tic_ids = set(results_df['tic_id'])
        tois_in_sample = toi_df[toi_df['tid'].isin(sample_tic_ids)].copy()
        
        # ベースライン内に最低2回のトランジットが観測できる周期を完備性評価の対象とする
        # (P < SECTOR_BASELINE_DAYS / (MIN_TRANSITS - 1) = 27.4 / 1 = 27.4 日)
        detectable_tois = tois_in_sample[tois_in_sample['pl_orbper'] < MAX_DETECTABLE_PERIOD_DAYS].copy()
        
        # 3. マッチング
        results_dict = results_df.set_index('tic_id').to_dict('index')
        matches = []
        for _, toi in detectable_tois.iterrows():
            tid = toi['tid']
            if tid in results_dict:
                det = results_dict[tid]
                t0_true = toi['pl_tranmid']
                if t0_true > 2450000: t0_true -= 2457000
                
                res = match_candidate(
                    p_detected=det['period'], t0_detected=det['t0'],
                    p_true=toi['pl_orbper'], t0_true=t0_true,
                    p_tol=self.p_tol, t0_tol=self.t0_tol, require_t0=self.require_t0
                )
                res.update({'tic_id': tid, 'det_power': det['power']})
                matches.append(res)
        
        matches_df = pd.DataFrame(matches)
        recovered_count = matches_df['is_match'].sum() if not matches_df.empty else 0
        
        # 4. 新候補 (Non-TOI, Non-EB)
        toi_ids = set(toi_df['tid'])
        def categorize(tic_id):
            if tic_id in toi_ids: return "TOI"
            if tic_id in eb_ids: return "EB"
            return "None"
        
        significant_sigs['type'] = significant_sigs['tic_id'].apply(categorize)
        new_candidates = significant_sigs[significant_sigs['type'] == "None"].sort_values('power', ascending=False)
        
        # 5. サマリーの構築
        summary = {
            'total_targets': len(results_df),
            'detectable_toi': len(detectable_tois),
            'recovered_toi': int(recovered_count),
            'completeness': recovered_count / len(detectable_tois) if len(detectable_tois) > 0 else 0,
            'significant_sigs': len(significant_sigs),
            'eb_found': (significant_sigs['type'] == "EB").sum(),
            'new_candidates': len(new_candidates),
            'fpr': len(new_candidates) / len(significant_sigs) if len(significant_sigs) > 0 else 0,
            'config': self.config['validation']
        }
        
        return {
            'summary': summary,
            'matches': matches_df,
            'new_candidates': new_candidates
        }
