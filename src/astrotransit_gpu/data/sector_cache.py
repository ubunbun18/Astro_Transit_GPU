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
            # TESS FITS specific columns
            time = data['TIME']
            flux = data['PDCSAP_FLUX']
            quality = data['QUALITY']
            
            # Basic cleaning: NaN removal and Quality masking
            mask = np.isfinite(time) & np.isfinite(flux) & (quality == 0)
            t = time[mask]
            f = flux[mask]
            
            if len(t) < 100:
                return {"status": "error", "error": "Too few points"}
            
            # Normalization
            f_med = np.median(f)
            f = f / f_med - 1.0
            
            # Simplified TIC ID extraction
            tic_id = int(hdul[0].header.get('TICID', 0))
            
            return {
                "tic_id": tic_id,
                "time": t.astype(np.float32),
                "flux": f.astype(np.float32),
                "flux_err": np.ones_like(f, dtype=np.float32) * 1e-4, # Simplified for speed
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
        
        valid_results = 0
        for res in results:
            if res['status'] == "ok":
                all_tic_ids.append(res['tic_id'])
                all_times.append(res['time'])
                all_fluxes.append(res['flux'])
                all_errs.append(res['flux_err'])
                offsets.append(offsets[-1] + len(res['time']))
                valid_results += 1
        
        print(f"Saving consolidated cache ({valid_results} targets)...")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Save heavy arrays in NPZ
        np.savez_compressed(
            self.data_path,
            time=np.concatenate(all_times),
            flux=np.concatenate(all_fluxes),
            flux_err=np.concatenate(all_errs),
            offsets=np.array(offsets, dtype=np.int64),
            tic_ids=np.array(all_tic_ids, dtype=np.int64)
        )
        
        # Save metadata for easy lookup
        meta_df = pd.DataFrame({
            "tic_id": all_tic_ids,
            "n_points": [len(t) for t in all_times]
        })
        meta_df.to_csv(self.meta_path, index=False)
        print(f"Cache built: {self.data_path}")

    def load(self):
        """Load the cache into memory."""
        data = np.load(self.data_path)
        return {
            "time": data['time'],
            "flux": data['flux'],
            "flux_err": data['flux_err'],
            "offsets": data['offsets'],
            "tic_ids": data['tic_ids']
        }
