import os
import numpy as np
import pandas as pd
import lightkurve as lk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from ..preprocess.clean import clean_lightcurve, to_arrays

def _process_one_fits(fits_path):
    """Worker function to read and clean one FITS file (optimized raw read)."""
    from astropy.io import fits
    import numpy as np
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            cols = hdul[1].columns.names
            
            # TESS FITS specific columns - Try QLP first then SPOC
            time = data['TIME']
            
            if 'KSPSAP_FLUX' in cols:
                flux = data['KSPSAP_FLUX']
                flux_err = data['KSPSAP_FLUX_ERR']
            elif 'PDCSAP_FLUX' in cols:
                flux = data['PDCSAP_FLUX']
                flux_err = data['PDCSAP_FLUX_ERR']
            elif 'SAP_FLUX' in cols:
                flux = data['SAP_FLUX']
                flux_err = data.get('SAP_FLUX_ERR', np.ones_like(flux) * 0.01)
            else:
                return {"status": "error", "error": f"No flux column found. Available: {cols}"}
                
            quality = data['QUALITY'] if 'QUALITY' in cols else np.zeros_like(time)
            
            # Basic cleaning: NaN removal and Quality masking
            mask = np.isfinite(time) & np.isfinite(flux) & (quality == 0)
            t = time[mask]
            f = flux[mask]
            fe = flux_err[mask]
            
            if len(t) < 100:
                return {"status": "error", "error": f"Too few points ({len(t)})"}
            
            # Normalization (if not already normalized, though QLP often is)
            f_med = np.median(f)
            if f_med == 0: return {"status": "error", "error": "Median flux is zero"}
            f_norm = f / f_med - 1.0
            fe_norm = fe / f_med
            
            # TIC ID extraction from header
            tic_id = int(hdul[0].header.get('TICID', 0))
            if tic_id == 0: # Try alternative header keys
                tic_id = int(hdul[0].header.get('OBJECT', 0))
            
            return {
                "tic_id": tic_id,
                "time": t.astype(np.float32),
                "flux": f_norm.astype(np.float32),
                "flux_err": fe_norm.astype(np.float32),
                "status": "ok"
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}

class SectorCache:
    """Consolidated binary cache for a TESS sector to enable high-speed GPU screening."""
    
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.data_path = os.path.join(cache_dir, "data.npz")
        self.meta_path = os.path.join(cache_dir, "metadata.csv")

    def build(self, fits_dir, workers=8):
        """Build the cache from a directory of FITS files."""
        import glob
        fits_files = glob.glob(os.path.join(fits_dir, "*.fits"))
        print(f"Building Sector Cache from {len(fits_files)} files...")
        
        all_tic_ids = []
        all_times = []
        all_fluxes = []
        all_errs = []
        offsets = [0]
        
        # Use ProcessPoolExecutor for CPU-intensive FITS parsing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(_process_one_fits, fits_files), 
                                total=len(fits_files), desc="Parsing FITS"))
        
        # V23/V24: Enforce uniform length (N_DATA=1312) for massive vectorized kernels
        TARGET_LEN = 1312
        
        valid_results = 0
        all_tic_ids = []
        all_fluxes = np.zeros((len(results), TARGET_LEN), dtype=np.float32)
        all_errs = np.zeros((len(results), TARGET_LEN), dtype=np.float32)
        
        # We'll use a common time array for the cache (FFI data is synchronized)
        common_time = np.zeros(TARGET_LEN, dtype=np.float32)
        time_set = False

        for i, res in enumerate(results):
            if res['status'] == "ok":
                tic_id = res['tic_id']
                t = res['time']
                f = res['flux']
                fe = res['flux_err']
                
                # Pad or trim to TARGET_LEN
                curr_len = len(t)
                if curr_len >= TARGET_LEN:
                    f_final = f[:TARGET_LEN]
                    fe_final = fe[:TARGET_LEN]
                    if not time_set:
                        common_time = t[:TARGET_LEN]
                        time_set = True
                else:
                    f_final = np.pad(f, (0, TARGET_LEN - curr_len), mode='constant')
                    fe_final = np.pad(fe, (0, TARGET_LEN - curr_len), mode='constant', constant_values=1.0)
                    if not time_set:
                        # Best effort: pad time with cadence spacing
                        dt = np.median(np.diff(t))
                        t_pad = t[-1] + np.arange(1, TARGET_LEN - curr_len + 1) * dt
                        common_time = np.concatenate([t, t_pad])
                        time_set = True
                
                all_tic_ids.append(tic_id)
                all_fluxes[valid_results] = f_final
                all_errs[valid_results] = fe_final
                valid_results += 1
        
        # Trim arrays to actual valid count
        all_fluxes = all_fluxes[:valid_results]
        all_errs = all_errs[:valid_results]
        
        print(f"Saving consolidated cache ({valid_results} targets, padded to {TARGET_LEN})...")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Save heavy arrays in NPZ
        np.savez(
            self.data_path,
            time=common_time, # Single common time array
            flux=all_fluxes,   # (N, 1312) matrix
            flux_err=all_errs, # (N, 1312) matrix
            tic_ids=np.array(all_tic_ids, dtype=np.int64),
            is_vectorized=True # Metadata flag
        )
        
        # Save metadata for easy lookup
        meta_df = pd.DataFrame({
            "tic_id": all_tic_ids,
            "n_points": [TARGET_LEN] * valid_results
        })
        meta_df.to_csv(self.meta_path, index=False)
        print(f"Cache built: {self.data_path}")

    def load(self):
        """Load the cache into memory."""
        data = np.load(self.data_path, allow_pickle=True)
        res = {
            "time": data['time'],
            "flux": data['flux'],
            "flux_err": data['flux_err'],
            "tic_ids": data['tic_ids'],
            "is_vectorized": bool(data.get('is_vectorized', False))
        }
        if not res['is_vectorized']:
            res['offsets'] = data['offsets']
        return res
