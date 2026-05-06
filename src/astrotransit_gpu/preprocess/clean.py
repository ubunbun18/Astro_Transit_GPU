import numpy as np
import lightkurve as lk

def clean_lightcurve(lc, sigma=5.0, window_length=401):
    """
    Standard cleaning pipeline for a light curve.
    1. Remove NaNs
    2. Quality mask filtering (already mostly done by Lightkurve download)
    3. Normalize
    4. Sigma clipping
    5. Flatten (detrending)
    """
    # 1. Remove NaNs
    lc = lc.remove_nans()
    
    # 2. Normalize
    lc = lc.normalize()
    
    # 3. Sigma clipping
    lc = lc.remove_outliers(sigma=sigma)
    
    # 4. Flatten
    lc_flattened = lc.flatten(window_length=window_length)
    
    return lc_flattened

def to_arrays(lc):
    """Convert lightkurve object to numpy arrays of (time, flux)."""
    return lc.time.value, lc.flux.value
