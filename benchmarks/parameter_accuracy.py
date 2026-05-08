import numpy as np
import pandas as pd
from astrotransit_gpu.data.exoplanet_archive import ExoplanetArchiveClient

def run_parameter_accuracy_analysis():
    # 1. データのロード
    results_df = pd.read_csv("outputs/bench_219331_v39.csv")
    results_df['tic_id'] = results_df['tic_id'].astype(str)
    
    toi_df = ExoplanetArchiveClient.get_toi_table()
    toi_df['tid'] = toi_df['tid'].astype(str)
    
    # 2. 回収成功天体の特定
    # 簡易化のため、large_scale_bench.py のロジックを再現
    recovered_stats = []
    
    print("Calculating Parameter Recovery Accuracy...")
    
    for _, toi in toi_df.iterrows():
        tid = toi['tid']
        res_row = results_df[results_df['tic_id'] == tid]
        
        if not res_row.empty:
            det = res_row.iloc[0]
            true_p = toi['pl_orbper']
            
            # 周期一致確認 (2%)
            if abs(det['period'] - true_p) / true_p < 0.02:
                # T0誤差の計算 (hours)
                t0_det = det['t0']
                t0_true = toi['pl_tranmid']
                if t0_true > 2450000: t0_true -= 2457000
                
                # 最近傍トランジットとの差
                n_cycles = round((t0_det - t0_true) / true_p)
                t0_error_days = abs(t0_det - t0_true - n_cycles * true_p)
                t0_error_min = t0_error_days * 24 * 60
                
                # 深さ誤差 (%)
                true_depth = toi['pl_trandep'] # ppm
                det_depth = det['depth'] * 1e6 # ppm
                depth_rel_err = abs(det_depth - true_depth) / true_depth * 100 if true_depth > 0 else 0
                
                # 継続時間誤差 (%)
                true_dur = toi['pl_trandurh'] # hours
                det_dur = det['duration'] # hours
                dur_rel_err = abs(det_dur - true_dur) / true_dur * 100 if true_dur > 0 else 0
                
                recovered_stats.append({
                    'tic_id': tid,
                    't0_err_min': t0_error_min,
                    'depth_err_pct': depth_rel_err,
                    'dur_err_pct': dur_rel_err
                })
                
    accuracy_df = pd.DataFrame(recovered_stats)
    
    print("\n# Priority 2: Parameter Recovery Accuracy")
    if not accuracy_df.empty:
        print(f"| Metric | Median Error | 95th Percentile | Unit |")
        print(f"| :--- | :--- | :--- | :--- |")
        print(f"| Transit Center (T0) | {accuracy_df['t0_err_min'].median():.2f} | {accuracy_df['t0_err_min'].quantile(0.95):.2f} | minutes |")
        print(f"| Transit Depth | {accuracy_df['depth_err_pct'].median():.2f} | {accuracy_df['depth_err_pct'].quantile(0.95):.2f} | % |")
        print(f"| Duration | {accuracy_df['dur_err_pct'].median():.2f} | {accuracy_df['dur_err_pct'].quantile(0.95):.2f} | % |")

if __name__ == "__main__":
    run_parameter_accuracy_analysis()
