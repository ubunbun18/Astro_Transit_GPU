import time
import numpy as np
from astrotransit_gpu.data.sector_cache import SectorCache
from astrotransit_gpu.search.screener import GpuScreener

def profile_computational_cost():
    sector = 1
    cache_dir = f'data/cache/s{sector:04d}'
    
    phases = {}
    
    # 1. データロード
    start = time.time()
    cache = SectorCache(cache_dir=cache_dir)
    sector_data = cache.load()
    phases['Data Loading (Cache)'] = time.time() - start
    
    # 2. 前処理 (GpuScreener内部で行われるGPU転送と重み計算)
    periods = np.linspace(0.5, 13.7, 100000)
    durations = np.linspace(0.05, 0.2, 10)
    screener = GpuScreener(periods, durations)
    
    # サンプル数を制限してプロファイル (1000天体)
    sub_data = {k: v[:1000] if isinstance(v, np.ndarray) and len(v.shape) > 0 else v for k, v in sector_data.items()}
    
    start = time.time()
    # screen_sector_vblsの内部を模倣
    import cupy as cp
    flux_matrix = cp.asarray(sub_data['flux'], dtype=cp.float32)
    err_matrix = cp.asarray(sub_data['flux_err'], dtype=cp.float32)
    weights_matrix = 1.0 / (err_matrix**2)
    weights_matrix[err_matrix > 0.9] = 0.0
    phases['Preprocessing & GPU Transfer'] = time.time() - start
    
    # 3. GPUカーネル実行 (V39)
    from astrotransit_gpu.search.vbls import run_vbls_massive
    start = time.time()
    _ = run_vbls_massive(
        sub_data['time'], flux_matrix, screener.periods, screener.durations,
        weights_matrix=weights_matrix, n_bins=200, use_blackwell=True
    )
    phases['GPU Kernel (V39 vBLS)'] = time.time() - start
    
    # 4. 後処理
    start = time.time()
    # 実際の結果回収などを想定
    _ = [float(x) for x in range(1000)]
    phases['Post-processing (CPU)'] = time.time() - start
    
    total = sum(phases.values())
    
    print("# Priority 8: Computational Cost Breakdown")
    print("| Phase | Time (s) | Percentage |")
    print("| :--- | :--- | :--- |")
    for phase, t in phases.items():
        print(f"| {phase} | {t:.4f} | {t/total:.1%} |")
        
    print(f"\nTotal Profiled Time: {total:.4f}s")
    print("\n✅ カーネル実行が支配的であれば、GPUの最適化が性能に直結していることが証明されます。")

if __name__ == "__main__":
    profile_computational_cost()
