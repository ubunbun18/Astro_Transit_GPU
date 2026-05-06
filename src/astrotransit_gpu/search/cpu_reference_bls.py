from astropy.timeseries import BoxLeastSquares
import numpy as np

def run_astropy_bls(time, flux, period_min=0.5, period_max=20.0, durations=None):
    """
    Run Astropy BLS as a reference.
    """
    if durations is None:
        durations = np.linspace(0.05, 0.2, 5) # hours to days? No, astropy expects days if time is in days.
        # Standard duration range is roughly 0.01 to 0.5 days.
    
    model = BoxLeastSquares(time, flux)
    results = model.autopower(durations, minimum_period=period_min, maximum_period=period_max)
    
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
        "power": results.power,
        "periods": results.period,
        "snr": results.depth_snr[best_idx]
    }
