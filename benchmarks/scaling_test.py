import time
import pandas as pd
import numpy as np
from astrotransit_gpu.search.screener import GpuScreener
from astrotransit_gpu.data.sector_cache import SectorCache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScalingTest")

def run_scaling_test():
    sector = 1
    # Standard search grid
    periods = np.linspace(0.5, 13.7, 100000)
    durations = np.linspace(0.05, 0.2, 10)
    
    screener = GpuScreener(periods, durations)
    cache = SectorCache(cache_dir=f'data/cache/s{sector:04d}')
    sector_data = cache.load()
    
    sample_sizes = [1000, 5000, 10000, 15881]
    results = []

    print("# Computational Scaling Test")
    print("| Targets | Total Time (s) | Time per Target (ms) |")
    print("| :--- | :--- | :--- |")

    for n in sample_sizes:
        # データのサブセットを作成
        sub_data = {
            'tic_ids': sector_data['tic_ids'][:n],
            'time': sector_data['time'],
            'flux': sector_data['flux'][:n],
            'flux_err': sector_data['flux_err'][:n],
            'is_vectorized': True
        }
        
        start_time = time.time()
        # V39カーネルを使用して解析
        _ = screener.screen_sector_vbls(sub_data, use_blackwell=True)
        elapsed = time.time() - start_time
        
        ms_per_target = (elapsed / n) * 1000
        results.append({
            'targets': n,
            'time': elapsed,
            'ms_per_target': ms_per_target
        })
        print(f"| {n:,} | {elapsed:.2f} | {ms_per_target:.2f} |")

    # 線形性の確認
    times = [r['time'] for r in results]
    targets = [r['targets'] for r in results]
    correlation = np.corrcoef(targets, times)[0, 1]
    
    print(f"\n**Linearity (Correlation Coefficient)**: **{correlation:.5f}**")
    if correlation > 0.99:
        print("✅ パイプラインは完全な線形スケーリングを示しており、設計は健全です。")
    else:
        print("⚠️ わずかな非線形性が検出されました（I/Oまたは初期化オーバーヘッドの可能性）。")

if __name__ == "__main__":
    run_scaling_test()
