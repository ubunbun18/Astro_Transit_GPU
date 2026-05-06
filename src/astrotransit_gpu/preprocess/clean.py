import numpy as np
import lightkurve as lk

def clean_lightcurve(lc, sigma=5.0, window_length=401, normalize=True):
    """
    Standard cleaning pipeline for a light curve.
    """
    lc = lc.remove_nans()
    
    if normalize:
        lc = lc.normalize()
    
    # 3. Sigma clipping
    lc = lc.remove_outliers(sigma=sigma)
    
    # 4. Flatten
    if window_length > 0:
        lc = lc.flatten(window_length=window_length)
    
    return lc

def to_arrays(lc):
    """Convert lightkurve object to numpy arrays of (time, flux)."""
    return lc.time.value, lc.flux.value
