import pandas as pd
import logging
from tqdm import tqdm
from ..data.lightkurve_client import LightkurveClient
from ..preprocess.clean import clean_lightcurve, to_arrays
from ..search.gpu_bls import run_gpu_bls
from ..validate.match import match_candidate
import numpy as np

logger = logging.getLogger(__name__)

def run_batch_known_search(targets, periods=None, durations=None):
    """
    Run transit search for a list of known targets and evaluate recovery.
    """
    client = LightkurveClient()
    if periods is None:
        periods = np.linspace(0.5, 20.0, 5000)
    if durations is None:
        durations = np.linspace(0.01, 0.2, 5)
        
    results = []
    
    for target in tqdm(targets, desc="Processing batch"):
        target_id = target['target_id']
        try:
            # 1. Download
            lc = client.download_lightcurve(target_id)
            # 2. Preprocess
            lc_clean = clean_lightcurve(lc)
            time_arr, flux_arr = to_arrays(lc_clean)
            
            # 3. Search
            search_res = run_gpu_bls(time_arr, flux_arr, periods, durations)
            
            # 4. Validate
            match = match_candidate(
                search_res['best_period'], 
                search_res['best_t0'], 
                target['period'], 
                target['t0']
            )
            
            # 5. Collect result
            results.append({
                'target_id': target_id,
                'true_period': target['period'],
                'detected_period': search_res['best_period'],
                'is_recovered': match['is_match'],
                'match_type': match['match_type'],
                'p_diff': match['p_diff'],
                'snr': search_res['snr']
            })
            
        except Exception as e:
            logger.error(f"Failed to process {target_id}: {e}")
            results.append({
                'target_id': target_id,
                'true_period': target['period'],
                'is_recovered': False,
                'error': str(e)
            })
            
    return pd.DataFrame(results)

def generate_batch_report_md(df, output_path):
    """Generate a Markdown summary of the batch run."""
    n_total = len(df)
    n_recovered = df['is_recovered'].sum()
    recovery_rate = n_recovered / n_total if n_total > 0 else 0
    
    md = f"# AstroTransit-GPU Batch Search Summary\n\n"
    md += f"## Performance Metrics\n\n"
    md += f"| Metric | Value |\n"
    md += f"| --- | --- |\n"
    md += f"| Total Targets | {n_total} |\n"
    md += f"| Recovered | {n_recovered} |\n"
    md += f"| Recovery Rate | {recovery_rate:.1%} |\n\n"
    
    md += "## Detailed Results\n\n"
    md += df[['target_id', 'true_period', 'detected_period', 'is_recovered', 'match_type', 'snr']].to_markdown(index=False)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    return md
