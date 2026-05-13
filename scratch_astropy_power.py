import numpy as np
import cupy as cp
from astropy.timeseries import BoxLeastSquares

def test_astropy_power_formulation():
    np.random.seed(42)
    time = np.linspace(0, 10, 1000)
    flux = np.random.normal(0, 0.01, size=1000)
    dy = np.ones_like(flux) * 0.01
    
    weights = 1.0 / (dy**2)
    w_sum = np.sum(weights)
    w_mean = np.sum(flux * weights) / w_sum
    flux -= w_mean
    
    # Run Astropy BLS
    model = BoxLeastSquares(time, flux, dy=dy)
    periods = np.array([3.0])
    durations = np.array([0.1])
    res = model.power(periods, durations)
    
    ap_power = res.power[0]
    ap_depth = res.depth[0]
    ap_snr = res.depth_snr[0]
    
    print(f"Astropy Power: {ap_power}")
    print(f"Astropy Depth: {ap_depth}")
    print(f"Astropy SNR:   {ap_snr}")
    
    # Replicate Astropy Power manually
    # Astropy calculates power = (sum(w*f))^2 / (sum(w) * (1 - r) * r) ... actually let's see.
    # In GPU V41: delta = cur_w * total_wf - cur_wf * total_w
    # SNR^2 = delta^2 / (cur_w * out_w * total_w)
    
    # We will test this relation:
    print(f"Astropy Power / SNR^2 = {ap_power / (ap_snr**2)}")
    
test_astropy_power_formulation()
