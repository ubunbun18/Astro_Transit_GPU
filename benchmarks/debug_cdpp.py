import lightkurve as lk
import numpy as np
import pandas as pd
from astrotransit_gpu.data.sector_cache import SectorCache

def debug_cdpp():
    sector = 1
    cache = SectorCache(cache_dir=f'data/cache/s{sector:04d}')
    sector_data = cache.load()
    
    # 最初のターゲットでテスト
    idx = 0
    flux = sector_data['flux'][idx]
    err = sector_data['flux_err'][idx]
    time = sector_data['time']
    
    mask = err < 0.9
    # lightkurveが期待する形式に調整 (相対的なppmではなく、1.0付近の絶対値)
    lc = lk.LightCurve(time=time[mask], flux=flux[mask] + 1.0)
    
    print(f"Time Range: {time[mask].min()} to {time[mask].max()}")
    print(f"Flux Mean: {lc.flux.mean()}, Std: {lc.flux.std()}")
    
    try:
        # 1時間のCDPP推定
        val = lc.estimate_cdpp(transit_duration=1/24.0)
        print(f"Estimated CDPP (1h): {val}")
    except Exception as e:
        print(f"CDPP Estimation Error: {e}")

if __name__ == "__main__":
    debug_cdpp()
