import pandas as pd
import numpy as np
from astrotransit_gpu.data.exoplanet_archive import ExoplanetArchiveClient

def debug_sector_2():
    # 1. データのロード
    results_df = pd.read_csv("outputs/bench_s2_v39.csv")
    results_df['tic_id'] = results_df['tic_id'].astype(str)
    
    toi_df = ExoplanetArchiveClient.get_toi_table()
    toi_df['tid'] = toi_df['tid'].astype(str)
    
    # 解析結果に含まれるTIC ID
    result_tids = set(results_df['tic_id'])
    
    # セクター2の検出可能TOI
    tois_in_sample = toi_df[toi_df['tid'].isin(result_tids)].copy()
    detectable_tois = tois_in_sample[tois_in_sample['pl_orbper'] < 13.7].copy()
    
    print(f"Debug Sector 2: {len(detectable_tois)} detectable TOIs in sample.")
    
    if len(detectable_tois) == 0:
        print("❌ No detectable TOIs found in the results TIC ID set!")
        return

    print("\n# Detailed Analysis of Sector 2 TOIs")
    print("| TIC ID | Period | Depth (ppm) | Power | Detected Period | Result |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for _, toi in detectable_tois.iterrows():
        tid = toi['tid']
        det = results_df[results_df['tic_id'] == tid].iloc[0]
        
        p_err = abs(det['period'] - toi['pl_orbper']) / toi['pl_orbper']
        is_match = (det['power'] > 7.1) and (p_err < 0.02)
        
        print(f"| {tid} | {toi['pl_orbper']:.2f} | {toi['pl_trandep']:.1f} | {det['power']:.1f} | {det['period']:.2f} | {'OK' if is_match else 'Miss'} |")

    # なぜMissなのかの統計
    print("\n# Miss Reasons")
    low_power = sum(results_df[results_df['tic_id'].isin(detectable_tois['tid'])]['power'] < 7.1)
    print(f"- Power < 7.1: {low_power}")
    
    shallow = sum(detectable_tois['pl_trandep'] < 500)
    print(f"- Depth < 500 ppm: {shallow}")

if __name__ == "__main__":
    debug_sector_2()
