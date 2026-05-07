import pandas as pd
import time
import os
import shutil
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from astrotransit_gpu.data.bulk_downloader import BulkDownloader

def benchmark():
    manifest_path = "data/bench_manifest.csv"
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found.")
        return

    df = pd.read_csv(manifest_path)
    # Use 100 files for each test to get a stable average
    sample_size = 100
    
    worker_options = [10, 20, 40, 80]
    results = []

    print(f"Starting Download Benchmark (Sample Size: {sample_size} files per test)")
    print("-" * 50)

    for workers in worker_options:
        test_dir = f"data/bench_test_{workers}"
        downloader = BulkDownloader(base_dir=test_dir, workers=workers)
        
        # Take a slice for this test
        test_df = df.sample(n=sample_size)
        
        print(f"Testing with {workers} workers...")
        start_time = time.time()
        downloader.download_from_manifest(test_df)
        end_time = time.time()
        
        duration = end_time - start_time
        speed = sample_size / duration
        print(f" -> Finished in {duration:.2f}s ({speed:.2f} files/sec)")
        
        results.append({
            "workers": workers,
            "duration": duration,
            "speed": speed
        })
        
        # Cleanup to free space and allow re-download
        shutil.rmtree(test_dir)

    print("\nBenchmark Results:")
    print("-" * 50)
    print("Workers | Time (s) | Speed (files/s)")
    for r in results:
        print(f"{r['workers']:7d} | {r['duration']:8.2f} | {r['speed']:14.2f}")
    
    best = max(results, key=lambda x: x['speed'])
    print("-" * 50)
    print(f"Optimal worker count: {best['workers']}")

if __name__ == "__main__":
    benchmark()
