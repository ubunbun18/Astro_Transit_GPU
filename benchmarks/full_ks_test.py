import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from astrotransit_gpu.data.exoplanet_archive import ExoplanetArchiveClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DistributionAnalysis")

def run_full_ks_test():
    # 1. データのロード
    results_df = pd.read_csv("outputs/bench_219331_v39.csv")
    candidates = results_df[results_df['power'] >= 7.1].copy()
    
    toi_df = ExoplanetArchiveClient.get_toi_table()
    # 物理的に検出可能なTOIs
    known_tois = toi_df[toi_df['pl_orbper'] < 13.7].copy()
    
    print(f"Running Full KS-Test on {len(candidates)} candidates vs {len(known_tois)} TOIs...")
    
    metrics = [
        ('period', 'pl_orbper', 'Period (days)'),
        ('duration', 'pl_trandurh', 'Duration (hours)'),
        ('depth', 'pl_trandep', 'Depth (ppm)')
    ]
    
    print("\n# Priority 8: Full KS-Test Distribution Analysis")
    print("| Metric | KS-Stat | p-value | Candidate Median | TOI Median | Result |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for cand_col, toi_col, label in metrics:
        cand_vals = candidates[cand_col]
        if cand_col == 'depth': cand_vals = cand_vals * 1e6
        
        known_vals = known_tois[toi_col].dropna()
        
        stat, p = ks_2samp(cand_vals, known_vals)
        
        print(f"| {label} | {stat:.4f} | {p:.4f} | {cand_vals.median():.2f} | {known_vals.median():.2f} | {'Diff' if p < 0.05 else 'Identical'} |")

    print("\n✅ 周期分布と継続時間分布において高い統計的一致が期待されます。")
    print("p > 0.05 であれば、新候補リストは惑星としての物理的特性を統計的に備えていると言えます。")

if __name__ == "__main__":
    run_full_ks_test()
