import numpy as np
import cupy as cp
from astropy.timeseries import BoxLeastSquares

def exact_astropy_power_math():
    np.random.seed(42)
    time = np.linspace(0, 10, 1000)
    flux = np.random.normal(0, 0.01, size=1000)
    dy = np.ones_like(flux) * 0.01
    
    weights = 1.0 / (dy**2)
    total_w = np.sum(weights)
    w_mean = np.sum(flux * weights) / total_w
    flux -= w_mean
    
    total_wf = np.sum(flux * weights) # should be ~0 now
    
    model = BoxLeastSquares(time, flux, dy=dy)
    periods = np.array([3.0])
    durations = np.array([0.1])
    res = model.power(periods, durations)
    
    # Let's find exactly the bins Astropy uses for this period and duration
    # Since we know Astropy just computes sums over 'in transit' vs 'out of transit':
    # Let's compute manually over the same phase
    best_t0 = res.transit_time[0]
    duration = res.duration[0]
    period = periods[0]
    
    phase = (time - best_t0 + 0.5 * duration) % period
    in_transit = (phase < duration)
    
    cur_w = np.sum(weights[in_transit])
    out_w = total_w - cur_w
    cur_wf = np.sum((flux * weights)[in_transit])
    
    # GPU V41 calculates SNR2:
    delta = cur_w * total_wf - cur_wf * total_w
    snr2 = delta**2 / (cur_w * out_w * total_w)
    depth = -delta / (cur_w * out_w)
    
    print(f"Manual depth: {depth}, Astropy depth: {res.depth[0]}")
    
    # Astropy Power formula from Kovacs 2002:
    # SR = (sum_in^2) / (w_in * (1 - w_in / sum_all))
    # Note cur_wf is sum_in (since flux is zero-meaned).
    astropy_power_manual = (cur_wf ** 2) / (cur_w * (1.0 - cur_w / total_w))
    print(f"Manual Astropy Power: {astropy_power_manual}")
    print(f"Actual Astropy Power: {res.power[0]}")
    
exact_astropy_power_math()
