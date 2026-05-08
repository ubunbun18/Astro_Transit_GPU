import numpy as np
import pandas as pd
import cupy as cp
import time
from astrotransit_gpu.search.screener import GpuScreener
from astrotransit_gpu.data.sector_cache import SectorCache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MassiveInjection")

def inject_transit(flux, time, period, depth_ppm, duration_hours, t0):
    depth = depth_ppm / 1e6
    duration_days = duration_hours / 24.0
    phase = (time - t0) % period
    half_dur = duration_days / 2.0
    in_transit = (phase < half_dur) | (phase > (period - half_dur))
    injected_flux = flux.copy()
    injected_flux[in_transit] -= depth
    return injected_flux

def run_massive_injection(n_total=10000):
    sector = 1
    cache = SectorCache(cache_dir=f'data/cache/s{sector:04d}')
    sector_data = cache.load()
    
    tic_ids = sector_data['tic_ids']
    times = sector_data['time']
    fluxes = sector_data['flux']
    errs = sector_data['flux_err']
    
    # 層化サンプリング用のビン定義
    p_bins = [(0.5, 2), (2, 5), (5, 10), (10, 13.7)]
    d_bins = [(500, 1000), (1000, 2500), (2500, 5000), (5000, 10000)]
    n_per_cell = n_total // (len(p_bins) * len(d_bins))
    
    injected_data = []
    
    print(f"Stratified Injection: {n_per_cell} samples per cell (Total ~10,000)...")
    
    # バッチ処理用配列
    all_p_inj = []
    all_d_inj = []
    all_t0_inj = []
    all_fluxes_inj = np.zeros((n_total, len(times)), dtype=np.float32)
    all_indices = []

    count = 0
    for p_range in p_bins:
        for d_range in d_bins:
            for _ in range(n_per_cell):
                if count >= n_total: break
                
                idx = np.random.randint(0, len(tic_ids))
                p = np.random.uniform(*p_range)
                d = np.random.uniform(*d_range)
                t0 = np.random.uniform(0, p)
                dur = 3.0 # hours (fixed for simplicity in scaling, or randomized)
                
                all_fluxes_inj[count] = inject_transit(fluxes[idx], times, p, d, dur, t0)
                all_p_inj.append(p)
                all_d_inj.append(d)
                all_t0_inj.append(t0)
                all_indices.append(idx)
                count += 1

    # 2. V39一括解析
    periods_grid = np.linspace(0.5, 13.7, 50000) # メモリ節約のため少しグリッドを粗く
    durations_grid = np.linspace(0.05, 0.2, 5)
    screener = GpuScreener(periods_grid, durations_grid)
    
    sub_data = {
        'tic_ids': tic_ids[all_indices],
        'time': times,
        'flux': all_fluxes_inj,
        'flux_err': errs[all_indices],
        'is_vectorized': True
    }
    
    print(f"Running massive analysis ({n_total} injections)...")
    start_time = time.time()
    results = screener.screen_sector_vbls(sub_data, use_blackwell=True)
    elapsed = time.time() - start_time
    
    # 3. 結果保存
    results_df = pd.DataFrame(results)
    results_df['p_inj'] = all_p_inj[:len(results)]
    results_df['depth_inj'] = all_d_inj[:len(results)]
    results_df['recovered'] = np.abs(results_df['period'] - results_df['p_inj']) / results_df['p_inj'] < 0.02
    
    results_df.to_csv("outputs/injection_results_10k.csv", index=False)
    print(f"Massive injection test complete in {elapsed:.2f}s. Results saved.")

if __name__ == "__main__":
    run_massive_injection(10000)
