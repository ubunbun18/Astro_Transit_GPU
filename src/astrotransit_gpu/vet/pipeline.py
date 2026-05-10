import os
import pandas as pd
import yaml
from .harmonics import group_harmonics
from .ranking import calculate_vetting_scores
from .catalog import apply_catalog_matching
from .plots import generate_top_plots
from .report import generate_html_report
from ..data.sector_cache import SectorCache

def run_vetting_pipeline(results_csv, cache_dir=None, config_path=None, out_dir="reports/vetting"):
    """
    Main entry point for the vetting pipeline.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load config
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
    # 2. Load results
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} candidates from {results_csv}")
    
    # 3. Filter status ok
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
        
    # 4. Apply Catalog Matching
    toi_path = config.get('toi_catalog')
    eb_path = config.get('eb_catalog')
    df = apply_catalog_matching(df, toi_path=toi_path, eb_path=eb_path)
    
    # 5. Group Harmonics
    print("Grouping harmonics...")
    df = group_harmonics(df, tolerance=config.get('harmonic_tolerance', 0.01))
    
    # 6. Calculate Vetting Scores
    print("Calculating vetting scores...")
    df = calculate_vetting_scores(df, config=config.get('scoring'))
    
    # 7. Sort and Rank
    df = df.sort_values('vetting_score', ascending=False)
    
    # 8. Save results
    ranked_csv = os.path.join(out_dir, "candidates_ranked.csv")
    df.to_csv(ranked_csv, index=False)
    print(f"Ranked candidates saved to {ranked_csv}")
    
    # 9. Generate Plots if cache is available
    if cache_dir and os.path.exists(cache_dir):
        print(f"Generating top plots using cache from {cache_dir}...")
        cache = SectorCache(cache_dir)
        try:
            sector_data = cache.load()
            plot_dir = os.path.join(out_dir, "plots")
            generate_top_plots(df, sector_data, plot_dir, top_n=config.get('top_n_plots', 50))
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")
            
    # 10. Generate Summary JSON & HTML Report
    meta = {
        "input_results": results_csv,
        "config_path": config_path,
        "kernel_version": "V39 Apex Predator"
    }
    
    from .report import save_summary_json, generate_html_report
    save_summary_json(df, out_dir, meta)
    generate_html_report(df, out_dir, meta)
            
    print(f"Vetting dashboard generated in {out_dir}")
    return df
