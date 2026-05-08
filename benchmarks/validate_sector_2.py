import os
import pandas as pd
import numpy as np
import logging
from astrotransit_gpu.data.sector_cache import SectorCache
from astrotransit_gpu.search.screener import GpuScreener
from astrotransit_gpu.data.exoplanet_archive import ExoplanetArchiveClient
from astrotransit_gpu.validate.match import match_candidate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiSectorValidation")

def run_sector_2_validation():
    sector = 2
    data_dir = f"data/tess_data/s{sector:04d}"
    cache_dir = f"data/cache/s{sector:04d}"
    
    # 1. キャッシュ構築
    print(f"\n--- Building Cache for Sector {sector} ---")
    cache = SectorCache(cache_dir=cache_dir)
    if not os.path.exists(cache.meta_path):
        cache.build(data_dir, workers=8)
    sector_data = cache.load()
    
    # 2. GPU解析 (V39)
    print(f"\n--- Running GPU Screening for Sector {sector} ---")
    periods = np.linspace(0.5, 13.7, 100000)
    durations = np.linspace(0.05, 0.2, 10)
    screener = GpuScreener(periods, durations)
    
    results = screener.screen_sector_vbls(sector_data, use_blackwell=True)
    
    # 結果保存
    results_df = pd.DataFrame(results)
    results_df['tic_id'] = results_df['tic_id'].astype(str)
    results_df.to_csv(f"outputs/bench_s2_v39.csv", index=False)
    
    # 3. 検証 (Completeness)
    print(f"\n--- Validating Results for Sector {sector} ---")
    toi_df = ExoplanetArchiveClient.get_toi_table()
    tic_ids = sector_data['tic_ids'].astype(str)
    
    # セクター2に含まれる物理的に検出可能なTOI
    tois_in_sample = toi_df[toi_df['tid'].astype(str).isin(tic_ids)].copy()
    detectable_tois = tois_in_sample[tois_in_sample['pl_orbper'] < 13.7].copy()
    
    recovered = 0
    results_dict = results_df.set_index('tic_id').to_dict('index')
    
    for _, toi in detectable_tois.iterrows():
        tid = str(toi['tid'])
        if tid in results_dict:
            det = results_dict[tid]
            t0_true = toi['pl_tranmid']
            if t0_true > 2450000: t0_true -= 2457000
            
            match = match_candidate(
                p_detected=det['period'], t0_detected=det['t0'],
                p_true=toi['pl_orbper'], t0_true=t0_true,
                p_tol=0.02, t0_tol=0.5, require_t0=False
            )
            if match['is_match']: recovered += 1
            
    n_total = len(detectable_tois)
    rate = recovered / n_total if n_total > 0 else 0
    
    print(f"\n# Multi-Sector Validation Result (Sector {sector})")
    print(f"- Total Targets: {len(tic_ids):,}")
    print(f"- Detectable TOIs: {n_total}")
    print(f"- Recovered TOIs: {recovered}")
    print(f"- Completeness: **{rate:.2%}**")
    
    print("\n✅ セクター2においてもセクター1と同様の完備性が確認されれば、パイプラインの汎用性が確定します。")

if __name__ == "__main__":
    run_sector_2_validation()
