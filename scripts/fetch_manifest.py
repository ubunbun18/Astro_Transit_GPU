import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from astrotransit_gpu.data.bulk_downloader import BulkDownloader

def fetch():
    print("Initializing BulkDownloader...")
    d = BulkDownloader('data/temp_bench')
    
    print("Querying MAST for Sector 1 QLP...")
    try:
        m = d.get_sector_manifest(1)
        if m.empty:
            print("Received empty manifest.")
            return
            
        print(f"Total products found: {len(m)}")
        m.to_csv('data/bench_manifest.csv', index=False)
        print("Saved manifest to data/bench_manifest.csv")
    except Exception as e:
        print(f"Error during query: {e}")

if __name__ == "__main__":
    fetch()
