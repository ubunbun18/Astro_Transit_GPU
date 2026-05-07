import pandas as pd
from tqdm import tqdm
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from astrotransit_gpu.data.lightkurve_client import LightkurveClient

def download_sector(csv_path="sector1_sample.csv", outdir=None, workers=4):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    target_ids = df['tic_id'].unique().tolist()
    
    print(f"Starting download of {len(target_ids)} targets from {csv_path}...")
    print(f"Saving to: {outdir if outdir else 'Default Lightkurve Cache'}")
    client = LightkurveClient(cache_dir=outdir)
    
    success = 0
    failed = 0
    
    # 実際には ThreadPoolExecutor を使って並列化すると速い
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def download_one(tid):
        try:
            client.download_lightcurve(str(tid))
            return True
        except Exception as e:
            return False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, tid): tid for tid in target_ids}
        for future in tqdm(as_completed(futures), total=len(target_ids), desc="Downloading"):
            if future.result():
                success += 1
            else:
                failed += 1
                
    print(f"\nDownload complete.")
    print(f"Total: {len(target_ids)}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="sector1_sample.csv")
    parser.add_argument("--outdir", type=str, default="data/tess_sector1")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    
    download_sector(args.csv, outdir=args.outdir, workers=args.workers)
