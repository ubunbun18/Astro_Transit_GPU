import numpy as np
import time
import os
import argparse

from astrotransit_gpu.search.screener import GpuScreener

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=None)
    parser.add_argument("--periods", type=int, default=100000)
    args = parser.parse_args()

    # Load Cache
    cache_path = 'data/cache/s0001/data.npz'
    if not os.path.exists(cache_path):
        print(f"Error: Cache not found at {cache_path}")
        return
        
    data = np.load(cache_path)
    n_targets_total = len(data['tic_ids'])
    
    # Slicing for subset benchmark
    n_targets = args.targets if args.targets else n_targets_total
    print(f"Loaded {n_targets_total} targets. Running benchmark on {n_targets} targets.")
    
    # Prepare sliced data dict
    data_bench = {
        'tic_ids': data['tic_ids'][:n_targets],
        'time': data['time'],
        'flux': data['flux'][:n_targets],
        'flux_err': data['flux_err'][:n_targets],
        'is_vectorized': True
    }
    
    # Grid
    periods = np.linspace(0.5, 15.0, args.periods)
    durations = np.linspace(0.01, 0.1, 16)
    
    screener = GpuScreener(periods, durations, n_bins=128)
    
    os.makedirs('outputs', exist_ok=True)
    output_path = f'outputs/bench_{n_targets}.csv'
    
    print(f"Starting Screening: {n_targets} stars x {len(periods)} periods...")
    start_time = time.time()
    
    # Use Blackwell V37 engine
    results = screener.screen_sector_vbls(
        data_bench, 
        output_path=output_path,
        target_batch_size=8000, 
        period_batch_size=25000,
        use_blackwell=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nBenchmark Complete for {n_targets} targets.")
    print(f"Total Elapsed: {elapsed:.2f}s")
    print(f"Throughput: {n_targets/elapsed:.2f} targets/sec")

if __name__ == "__main__":
    main()
