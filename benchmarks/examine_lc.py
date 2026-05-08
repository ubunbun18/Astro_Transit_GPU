import numpy as np
import os

def examine_target(tic_id):
    cache_path = "data/cache/s0001/data.npz"
    data = np.load(cache_path)
    
    tic_ids = data['tic_ids']
    idx = np.where(tic_ids == tic_id)[0]
    
    if len(idx) == 0:
        print(f"Target {tic_id} not found in cache.")
        return
    
    i = idx[0]
    flux = data['flux'][i]
    time = data['time']
    
    print(f"TIC {tic_id} stats:")
    print(f"  Valid points (non-zero): {np.count_nonzero(flux)}")
    print(f"  Mean flux: {np.mean(flux):.6f}")
    print(f"  RMS flux: {np.std(flux):.6f}")
    
    # Check for gaps
    gaps = np.where(flux == 0)[0]
    print(f"  Padded/Gap points: {len(gaps)} out of {len(flux)}")
    
    # Save a small snippet to check
    # print(flux[:20])

if __name__ == "__main__":
    examine_target(231670397) # TOI 101
