from astropy.timeseries import BoxLeastSquares
import numpy as np

def run_astropy_bls(time, flux, dy=None, periods=None, period_min=0.5, period_max=20.0, durations=None):
    """
    Run Astropy BLS as a reference.
    """
    if durations is None:
        durations = np.linspace(0.01, 0.2, 5) 
    
    model = BoxLeastSquares(time, flux, dy=dy)
    
    if periods is None:
        results = model.autopower(durations, minimum_period=period_min, maximum_period=period_max, objective="snr")
    else:
        results = model.power(periods, durations, objective="snr")
    
    best_idx = np.argmax(results.power)
    best_period = results.period[best_idx]
    best_t0 = results.transit_time[best_idx]
    best_duration = results.duration[best_idx]
    best_depth = results.depth[best_idx]
    
    return {
        "period": best_period,
        "t0": best_t0,
        "duration": best_duration,
        "depth": best_depth,
        "best_power": results.power[best_idx],
        "power": results.power,
        "snr": results.depth_snr[best_idx]
    }
