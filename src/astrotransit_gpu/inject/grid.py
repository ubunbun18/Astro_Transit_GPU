import numpy as np
import pandas as pd
from tqdm import tqdm
from .box import inject_box_transit
from ..search.gpu_bls import run_gpu_bls
from ..validate.match import match_candidate

def run_injection_recovery_experiment(time, flux, periods_to_inject, depths_to_inject, n_trials=5, search_periods=None):
    """
    Perform an injection/recovery experiment on a single light curve.
    
    Args:
        time (np.ndarray): Time array.
        flux (np.ndarray): Original flux array.
        periods_to_inject (list): List of periods to test.
        depths_to_inject (list): List of depths (relative) to test.
        n_trials (int): Number of random trials (random epochs) per grid cell.
        search_periods (np.ndarray): Grid of periods to search over.
        
    Returns:
        pd.DataFrame: Results of all trials.
    """
    if search_periods is None:
        search_periods = np.linspace(0.5, 20.0, 5000)
    
    search_durations = np.linspace(0.01, 0.2, 5)
    
    results = []
    
    total_iterations = len(periods_to_inject) * len(depths_to_inject) * n_trials
    pbar = tqdm(total=total_iterations, desc="Injection/Recovery Grid")
    
    for p_inj in periods_to_inject:
        for d_inj in depths_to_inject:
            for trial in range(n_trials):
                # 1. Randomize epoch
                t0_inj = time[0] + np.random.random() * p_inj
                dur_inj = 0.1 # Fixed for simplicity in grid, could be randomized
                
                # 2. Inject
                flux_injected = inject_box_transit(time, flux, p_inj, t0_inj, dur_inj, d_inj)
                
                # 3. Search
                search_res = run_gpu_bls(time, flux_injected, search_periods, search_durations)
                
                # 4. Validate
                match = match_candidate(search_res['best_period'], search_res['best_t0'], p_inj, t0_inj)
                
                # 5. Collect
                results.append({
                    'injected_period': p_inj,
                    'injected_depth': d_inj,
                    'trial': trial,
                    'is_recovered': match['is_match'],
                    'match_type': match['match_type'],
                    'detected_period': search_res['best_period'],
                    'snr': search_res['snr']
                })
                pbar.update(1)
                
    pbar.close()
    return pd.DataFrame(results)

def calculate_recovery_map(df):
    """Calculate recovery probability per grid cell."""
    pivot = df.groupby(['injected_period', 'injected_depth'])['is_recovered'].mean().unstack()
    return pivot
