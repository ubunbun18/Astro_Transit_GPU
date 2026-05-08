import time
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from astrotransit_gpu.search.screener import GpuScreener
from astrotransit_gpu.data.sector_cache import SectorCache
from astrotransit_gpu.data.exoplanet_archive import ExoplanetArchiveClient
from astrotransit_gpu.validate.match import match_candidate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AstropyVsGpu")

def run_comparison():
    # 1. データの準備
    sector = 1
    cache = SectorCache(cache_dir=f'data/cache/s{sector:04d}')
    sector_data = cache.load()
    toi_df = ExoplanetArchiveClient.get_toi_table()
    
    # 2. 検出対象の選定 (P < 13.7d)
    tic_ids = sector_data['tic_ids']
    tois_in_sample = toi_df[toi_df['tid'].isin(tic_ids)].copy()
    detectable_tois = tois_in_sample[tois_in_sample['pl_orbper'] < 13.7].copy()
    
    target_tic_ids = detectable_tois['tid'].unique()[:50] # 時間短縮のため上位50件で比較
    print(f"Comparing GPU vs Astropy on {len(target_tic_ids)} targets...")
    
    # 3. 探索条件の統一
    periods = np.linspace(0.5, 13.7, 100000)
    durations = [0.1] # 計算時間短縮のため1種類に固定
    
    # --- GPU Part ---
    screener = GpuScreener(periods, durations)
    gpu_indices = [np.where(tic_ids == int(tid))[0][0] for tid in target_tic_ids]
    
    gpu_data = {
        'tic_ids': tic_ids[gpu_indices],
        'time': sector_data['time'],
        'flux': sector_data['flux'][gpu_indices],
        'flux_err': sector_data['flux_err'][gpu_indices],
        'is_vectorized': True
    }
    
    print("\n[GPU] Running analysis...")
    start_gpu = time.time()
    gpu_results = screener.screen_sector_vbls(gpu_data, use_blackwell=True)
    end_gpu = time.time()
    
    # --- Astropy Part ---
    print("\n[Astropy] Running analysis (this may take a while)...")
    astropy_results = []
    start_astropy = time.time()
    for i, tid in enumerate(target_tic_ids):
        t = sector_data['time']
        y = sector_data['flux'][gpu_indices[i]]
        dy = sector_data['flux_err'][gpu_indices[i]]
        
        # Astropy BLS
        model = BoxLeastSquares(t, y, dy)
        periodogram = model.power(periods, durations[0])
        
        best_idx = np.argmax(periodogram.power)
        astropy_results.append({
            'tic_id': tid,
            'period': float(periods[best_idx]),
            'power': float(periodogram.power[best_idx]),
            't0': float(periodogram.transit_time[best_idx])
        })
    end_astropy = time.time()
    
    # 4. 回収率の計算
    def compute_rate(results, truth_df):
        recovered = 0
        truth_df['tid'] = truth_df['tid'].astype(str)
        for res in results:
            tid_str = str(res['tic_id'])
            subset = truth_df[truth_df['tid'] == tid_str]
            if subset.empty:
                logger.warning(f"TIC {tid_str} not found in truth_df")
                continue
            
            truth = subset.iloc[0]
            t0_true = truth['pl_tranmid']
            if t0_true > 2450000: t0_true -= 2457000
            
            match = match_candidate(
                p_detected=res['period'], t0_detected=res['t0'],
                p_true=truth['pl_orbper'], t0_true=t0_true,
                p_tol=0.02, t0_tol=0.5, require_t0=True
            )
            if match['is_match']: recovered += 1
        return recovered / len(results)

    gpu_rate = compute_rate(gpu_results, detectable_tois)
    astropy_rate = compute_rate(astropy_results, detectable_tois)
    
    # 5. レポート
    print("\n# GPU vs Astropy Direct Comparison")
    print(f"| Metric | GPU (V39) | Astropy (Standard) |")
    print(f"| :--- | :--- | :--- |")
    print(f"| Total Time | **{end_gpu - start_gpu:.2f} s** | {end_astropy - start_astropy:.2f} s |")
    print(f"| Throughput | **{len(target_tic_ids)/(end_gpu - start_gpu):.2f} targets/s** | {len(target_tic_ids)/(end_astropy - start_astropy):.2f} targets/s |")
    print(f"| Completeness | **{gpu_rate:.2%}** | {astropy_rate:.2%} |")
    print(f"| Speedup | **{ (end_astropy - start_astropy) / (end_gpu - start_gpu) :.1f}x** | 1.0x |")

if __name__ == "__main__":
    run_comparison()
