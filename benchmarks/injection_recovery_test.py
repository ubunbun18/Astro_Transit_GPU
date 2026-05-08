import numpy as np
import pandas as pd
import cupy as cp
import time
from astrotransit_gpu.search.screener import GpuScreener
from astrotransit_gpu.data.sector_cache import SectorCache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InjectionRecovery")

def inject_transit(flux, time, period, depth_ppm, duration_hours, t0):
    """光度曲線にボックス型トランジットを注入"""
    depth = depth_ppm / 1e6
    duration_days = duration_hours / 24.0
    
    # フェーズ計算
    phase = (time - t0) % period
    # トランジット内判定 (中心を0に想定)
    half_dur = duration_days / 2.0
    in_transit = (phase < half_dur) | (phase > (period - half_dur))
    
    injected_flux = flux.copy()
    injected_flux[in_transit] -= depth
    return injected_flux

def run_injection_recovery(n_injections=1000):
    sector = 1
    cache = SectorCache(cache_dir=f'data/cache/s{sector:04d}')
    sector_data = cache.load()
    
    tic_ids = sector_data['tic_ids']
    times = sector_data['time']
    fluxes = sector_data['flux']
    errs = sector_data['flux_err']
    
    # 1. ランダムサンプリング
    indices = np.random.choice(len(tic_ids), n_injections, replace=False)
    
    # 注入パラメータの生成
    p_inj = np.random.uniform(0.5, 13.7, n_injections)
    depth_inj = np.random.uniform(500, 10000, n_injections) # ppm
    dur_inj = np.random.uniform(1, 5, n_injections) # hours
    t0_inj = np.random.uniform(0, p_inj, n_injections)
    
    injected_fluxes = np.zeros((n_injections, len(times)), dtype=np.float32)
    
    print(f"Injecting {n_injections} signals...")
    for i in range(n_injections):
        idx = indices[i]
        injected_fluxes[i] = inject_transit(
            fluxes[idx], times, p_inj[i], depth_inj[i], dur_inj[i], t0_inj[i]
        )
    
    # 2. V39一括解析
    periods_grid = np.linspace(0.5, 13.7, 100000)
    durations_grid = np.linspace(0.05, 0.2, 10)
    screener = GpuScreener(periods_grid, durations_grid)
    
    sub_data = {
        'tic_ids': tic_ids[indices],
        'time': times,
        'flux': injected_fluxes,
        'flux_err': errs[indices],
        'is_vectorized': True
    }
    
    print("Running batch analysis on GPU...")
    start_time = time.time()
    results = screener.screen_sector_vbls(sub_data, use_blackwell=True)
    elapsed = time.time() - start_time
    
    # 3. 回収判定
    results_df = pd.DataFrame(results)
    results_df['p_inj'] = p_inj
    results_df['depth_inj'] = depth_inj
    
    # 周期誤差 2% 以内を回収成功とする
    results_df['recovered'] = np.abs(results_df['period'] - p_inj) / p_inj < 0.02
    
    recovery_rate = results_df['recovered'].mean()
    print(f"\n# Injection-Recovery Result")
    print(f"- Total Injections: {n_injections}")
    print(f"- Overall Recovery Rate: {recovery_rate:.2%}")
    print(f"- Analysis Time: {elapsed:.2f}s")
    
    # 結果の保存 (Sensitivity Map用)
    results_df.to_csv("outputs/injection_results.csv", index=False)
    print("Results saved to outputs/injection_results.csv")

if __name__ == "__main__":
    run_injection_recovery(1000)
