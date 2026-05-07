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
        
    def screen_sector(self, sector_data, output_path=None, batch_size=10000, use_blackwell=False):
        """
        Process an entire sector from SectorCache data.
        Automatically detects if data can be vectorized.
        """
        offsets = sector_data['offsets']
        tic_ids = sector_data['tic_ids']
        n_targets = len(tic_ids)
        
        # Check if all targets have the same length (common for FFI)
        lengths = np.diff(offsets)
        if len(np.unique(lengths)) == 1:
            return self.screen_sector_vbls(sector_data, output_path, batch_size=batch_size, use_blackwell=use_blackwell)
        else:
            # Fallback to sequential for ragged data
            return self._screen_sector_sequential(sector_data, output_path)

    def screen_sector_vbls(self, sector_data, output_path=None, target_batch_size=10000, period_batch_size=10000, use_blackwell=False):
        """Ultra-fast vectorized screening for uniform-length data (Scales to 1M targets)."""
        from .vbls import run_vbls_massive
        import csv
        
        tic_ids = sector_data['tic_ids']
        n_targets = len(tic_ids)
        
        if sector_data.get('is_vectorized', False):
            # V23/V24 Optimized Cache Format
            common_time = sector_data['time']
            flux_matrix = sector_data['flux']
            weights_matrix = 1.0 / (sector_data['flux_err']**2)
            n_pts = len(common_time)
        else:
            # Legacy/Ragged Format - Manual Reshape
            time_all = cp.asarray(sector_data['time'], dtype=self.dtype)
            flux_all = cp.asarray(sector_data['flux'], dtype=self.dtype)
            err_all = cp.asarray(sector_data['flux_err'], dtype=self.dtype)
            offsets = sector_data['offsets']
            n_pts = int(np.diff(offsets)[0])
            common_time = time_all[:n_pts]
            flux_matrix = flux_all.reshape(n_targets, n_pts)
            weights_matrix = (1.0 / (err_all * err_all)).reshape(n_targets, n_pts)
        
        # Prepare CSV file
        out_f = None
        writer = None
        if output_path:
            out_f = open(output_path, 'w', newline='')
            writer = csv.DictWriter(out_f, fieldnames=[
                "tic_id", "status", "period", "t0", "depth", "duration", "power", "n_points", "error"
            ])
            writer.writeheader()

        print(f"Screening {n_targets} targets and {len(self.periods)} periods using Massive Vectorized GPU BLS...")
        start_total = time.time()
        
        # Run massive vBLS

        # Run massive vBLS
        results_raw = run_vbls_massive(
            common_time, flux_matrix, self.periods, self.durations,
            weights_matrix=weights_matrix, n_bins=self.n_bins, dtype=self.dtype,
            target_batch_size=target_batch_size, period_batch_size=period_batch_size,
            use_blackwell=use_blackwell
        )
        
        # Save results
        final_results = []
        
        # Move all results to CPU at once for high-speed processing
        res_cpu = {k: v.get() if hasattr(v, 'get') else v for k, v in results_raw.items()}
        
        for i in range(n_targets):
            res_row = {
                "tic_id": int(tic_ids[i]),
                "period": float(res_cpu['best_period'][i]),
                "t0": float(res_cpu['best_t0'][i]),
                "depth": float(res_cpu['best_depth'][i]),
                "duration": float(res_cpu['best_duration'][i]),
                "power": float(res_cpu['snr'][i]),
                "n_points": n_pts,
                "status": "ok",
                "error": ""
            }
            final_results.append(res_row)
            if writer:
                writer.writerow(res_row)
        
        if out_f:
            out_f.close()

        end_total = time.time()
        elapsed = end_total - start_total
        print(f"\nScreening complete in {elapsed:.2f}s ({n_targets * len(self.periods) / elapsed / 1e9:.2f} G-searches/sec)")
        
        return final_results

    def _screen_sector_sequential(self, sector_data, output_path=None):
        """Original sequential screening logic."""
        import csv
        time_all = sector_data['time']
        flux_all = sector_data['flux']
        err_all = sector_data['flux_err']
        offsets = sector_data['offsets']
        tic_ids = sector_data['tic_ids']
        
        results = []
        n_targets = len(tic_ids)
        
        out_f = None
        writer = None
        if output_path:
            out_f = open(output_path, 'w', newline='')
            writer = csv.DictWriter(out_f, fieldnames=[
                "tic_id", "status", "period", "t0", "depth", "duration", "power", "n_points", "error"
            ])
            writer.writeheader()

        print(f"Screening {n_targets} targets sequentially on GPU...")
        start_total = time.time()
        
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
