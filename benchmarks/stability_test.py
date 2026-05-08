import numpy as np
import pandas as pd
import random
from astrotransit_gpu.validate.match import match_candidate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StabilityTest")

def compute_completeness(sample_tois, results_df):
    results_dict = results_df.set_index('tic_id').to_dict('index')
    recovered = 0
    for _, toi in sample_tois.iterrows():
        tid = str(toi['tid'])
        if tid in results_dict:
            det = results_dict[tid]
            t0_true = toi['pl_tranmid']
            if t0_true > 2450000: t0_true -= 2457000
            
            match = match_candidate(
                p_detected=det['period'], t0_detected=det['t0'],
                p_true=toi['pl_orbper'], t0_true=t0_true,
                p_tol=0.02, t0_tol=0.5, require_t0=True
            )
            if match['is_match']: recovered += 1
    return recovered / len(sample_tois) if len(sample_tois) > 0 else 0

def run_stability_tests():
    # データのロード
    from astrotransit_gpu.data.exoplanet_archive import ExoplanetArchiveClient
    results_df = pd.read_csv("outputs/bench_219331_v39.csv")
    results_df['tic_id'] = results_df['tic_id'].astype(str)
    
    toi_df = ExoplanetArchiveClient.get_toi_table()
    toi_df['tid'] = toi_df['tid'].astype(str)
    
    # 物理的に検出可能な156件
    detectable_tois = toi_df[toi_df['pl_orbper'] < 13.7].copy()
    # 実際にサンプルに含まれるものに限定
    sample_tic_ids = set(results_df['tic_id'])
    detectable_tois = detectable_tois[detectable_tois['tid'].isin(sample_tic_ids)]
    
    print(f"Running Stability Tests on {len(detectable_tois)} detectable TOIs...")
    
    # 1. Half-Sample Test
    print("\n# Half-Sample Stability Test")
    all_indices = detectable_tois.index.tolist()
    random.shuffle(all_indices)
    
    mid = len(all_indices) // 2
    group_a = detectable_tois.loc[all_indices[:mid]]
    group_b = detectable_tois.loc[all_indices[mid:]]
    
    comp_a = compute_completeness(group_a, results_df)
    comp_b = compute_completeness(group_b, results_df)
    
    diff = abs(comp_a - comp_b)
    print(f"| Group | Size | Completeness |")
    print(f"| :--- | :--- | :--- |")
    print(f"| A | {len(group_a)} | {comp_a:.2%} |")
    print(f"| B | {len(group_b)} | {comp_b:.2%} |")
    print(f"\n**Difference**: **{diff:.2%}**")
    
    if diff < 0.05:
        print("✅ ハーフサンプル間での完備性の差異は5%未満であり、検証結果は極めて安定しています。")
    else:
        print("⚠️ サンプル分割による有意な差異が認められます。統計的ゆらぎの可能性があります。")

if __name__ == "__main__":
    run_stability_tests()
