import time
import numpy as np
import cupy as cp
from tqdm import tqdm
from .gpu_bls import run_gpu_bls

class GpuScreener:
    """High-performance screening engine for entire sectors."""
    
    def __init__(self, periods, durations, n_bins=200, dtype=cp.float32):
        self.periods = cp.asarray(periods, dtype=dtype)
        self.durations = cp.asarray(durations, dtype=dtype)
        self.n_bins = n_bins
        self.dtype = dtype
        
        # Precompute constants to avoid redundant kernel work
        self.inv_periods = 1.0 / self.periods
        
    def screen_sector(self, sector_data, output_path=None):
        """
        Process an entire sector from SectorCache data.
        
        Args:
            sector_data (dict): Dict containing 'time', 'flux', 'flux_err', 'offsets', 'tic_ids'.
            output_path (str): Optional path to save results progressively.
        """
        import csv
        time_all = sector_data['time']
        flux_all = sector_data['flux']
        err_all = sector_data['flux_err']
        offsets = sector_data['offsets']
        tic_ids = sector_data['tic_ids']
        
        results = []
        n_targets = len(tic_ids)
        
        # Prepare CSV file
        out_f = None
        writer = None
        if output_path:
            out_f = open(output_path, 'w', newline='')
            writer = csv.DictWriter(out_f, fieldnames=[
                "tic_id", "status", "period", "t0", "depth", "duration", "power", "n_points", "error"
            ])
            writer.writeheader()

        print(f"Screening {n_targets} targets on GPU...")
        start_total = time.time()
        
        # Loop through targets
        for i in tqdm(range(n_targets), desc="Screening"):
            s, e = offsets[i], offsets[i+1]
            t = time_all[s:e]
            y = flux_all[s:e]
            dy = err_all[s:e]
            n_pts = len(t)
            
            try:
                raw_res = run_gpu_bls(
                    t, y, self.periods, self.durations,
                    flux_err=dy, n_bins=self.n_bins, dtype=self.dtype
                )
                res_row = {
                    "tic_id": int(tic_ids[i]),
                    "period": float(raw_res['best_period']),
                    "t0": float(raw_res['best_t0']),
                    "depth": float(raw_res['best_depth']),
                    "duration": float(raw_res['best_duration']),
                    "power": float(raw_res['snr']),
                    "n_points": n_pts,
                    "status": "ok",
                    "error": ""
                }
            except Exception as ex:
                res_row = {
                    "tic_id": int(tic_ids[i]),
                    "status": "error",
                    "n_points": n_pts,
                    "error": str(ex)
                }
            
            results.append(res_row)
            if writer:
                writer.writerow(res_row)
                out_f.flush()
                
        if out_f:
            out_f.close()

        end_total = time.time()
        elapsed = end_total - start_total
        print(f"\nScreening complete in {elapsed:.2f}s ({n_targets/elapsed:.2f} targets/sec)")
        
        return results
