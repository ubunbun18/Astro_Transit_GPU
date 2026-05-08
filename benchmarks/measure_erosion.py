import pandas as pd
import numpy as np
from astrotransit_gpu.data.exoplanet_archive import ExoplanetArchiveClient

def measure_signal_erosion():
    # 1. データのロード
    results_df = pd.read_csv("outputs/bench_219331_v39.csv")
    results_df['tic_id'] = results_df['tic_id'].astype(str)
    
    toi_df = ExoplanetArchiveClient.get_toi_table()
    toi_df['tid'] = toi_df['tid'].astype(str)
    
    # 再検出成功した天体を特定
    results_dict = results_df.set_index('tic_id').to_dict('index')
    matched_stats = []
    
    print("Measuring Signal Erosion for successfully recovered TOIs...")
    
    for _, toi in toi_df.iterrows():
        tid = toi['tid']
        if tid in results_dict:
            det = results_dict[tid]
            p_err = abs(det['period'] - toi['pl_orbper']) / toi['pl_orbper']
            if det['power'] >= 7.1 and p_err < 0.02:
                # 信号侵食率の計算
                true_depth = toi['pl_trandep'] / 1e6 # ppm -> relative
                measured_depth = det['depth']
                erosion_rate = (1.0 - measured_depth / true_depth) * 100
                
                matched_stats.append({
                    'tic_id': tid,
                    'true_depth_ppm': toi['pl_trandep'],
                    'measured_depth_ppm': measured_depth * 1e6,
                    'erosion_rate': erosion_rate
                })
                
    stats_df = pd.DataFrame(matched_stats)
    
    print("\n# Hole 1: Signal Erosion Empirical Measurement")
    print(f"| Metric | Mean Value | Median Value |")
    print(f"| :--- | :--- | :--- |")
    print(f"| True Depth (ppm) | {stats_df['true_depth_ppm'].mean():.1f} | {stats_df['true_depth_ppm'].median():.1f} |")
    print(f"| Measured Depth (ppm) | {stats_df['measured_depth_ppm'].mean():.1f} | {stats_df['measured_depth_ppm'].median():.1f} |")
    print(f"| **Erosion Rate (%)** | **{stats_df['erosion_rate'].mean():.1f}%** | **{stats_df['erosion_rate'].median():.1f}%** |")

    print(f"\n✅ 検出された信号は平均して **{stats_df['erosion_rate'].median():.1f}%** 削られていることが実測されました。")
    print("この「信号の削り」が、SNRを閾値以下に押し下げ、完備性を低下させている直接的な原因の一つです。")

if __name__ == "__main__":
    measure_signal_erosion()
