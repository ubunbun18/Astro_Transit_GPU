import numpy as np
import pandas as pd
import os

def identify_sector():
    cache_path = "data/cache/s0001/data.npz"
    if not os.path.exists(cache_path):
        print("Cache not found.")
        return
    
    data = np.load(cache_path)
    times = data['time']
    
    t_start = np.min(times[times > 0])
    t_end = np.max(times)
    
    print(f"Time Range in Cache (BTJD): {t_start:.4f} to {t_end:.4f}")
    print(f"Duration: {t_end - t_start:.2f} days")
    
    # TESS Sector lookup (Approximate BTJD ranges)
    # Sector 1: 1325 - 1353
    # Sector 2: 1354 - 1381
    # Sector 14: 1683 - 1710
    
    if 1320 < t_start < 1330:
        sector = 1
    elif 1350 < t_start < 1360:
        sector = 2
    elif 1680 < t_start < 1690:
        sector = 14
    else:
        sector = "Unknown"
        
    print(f"Detected Sector: {sector}")
    
    print("\nTOI Table Columns:")
    print(list(toi_df.columns))
    
    # Check if any TOIs have TIC ID in sample
    
    print(f"\nNumber of TOIs with TIC IDs in sample: {len(tois_in_sample)}")
    
    # Check TOI observation sectors if available (column 'sectors')
    if 'sectors' in tois_in_sample.columns:
        print("\nFirst 5 TOIs sectors list:")
        print(tois_in_sample[['tid', 'sectors']].head())
        # Check how many have the detected sector in their list
        sector_str = str(sector)
        matching_sector = tois_in_sample[tois_in_sample['sectors'].astype(str).str.contains(sector_str)]
        print(f"\nNumber of TOIs confirmed observed in Sector {sector}: {len(matching_sector)}")
    else:
        print("Sector column not found in TOI table. Checking individual famous targets...")
        
if __name__ == "__main__":
    identify_sector()
