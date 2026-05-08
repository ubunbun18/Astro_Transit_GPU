import numpy as np
import pandas as pd
import os
from astrotransit_gpu.data.exoplanet_archive import ExoplanetArchiveClient

def check_toi_columns():
    toi_df = ExoplanetArchiveClient.get_toi_table()
    print("TOI Table Columns:")
    print(toi_df.columns.tolist())
    
    # Check for any column that might contain sector info
    potential_cols = [c for c in toi_df.columns if 'sector' in c.lower()]
    print(f"\nPotential Sector Columns: {potential_cols}")
    
    if potential_cols:
        print("\nSample values for potential columns:")
        print(toi_df[potential_cols].dropna().head())

if __name__ == "__main__":
    check_toi_columns()
