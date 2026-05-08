import pandas as pd
import numpy as np
from astrotransit_gpu.data.exoplanet_archive import ExoplanetArchiveClient
from astrotransit_gpu.validate.match import match_candidate

def run_snr_threshold_analysis():
    # 1. データのロード
    results_df = pd.read_csv("outputs/bench_219331_v39.csv")
    results_df['tic_id'] = results_df['tic_id'].astype(str)
    
    toi_df = ExoplanetArchiveClient.get_toi_table()
    toi_df['tid'] = toi_df['tid'].astype(str)
    
    sample_tids = set(results_df['tic_id'])
    detectable_tois = toi_df[(toi_df['pl_orbper'] < 13.7) & (toi_df['tid'].isin(sample_tids))].copy()
    
    thresholds = np.arange(5.0, 15.5, 0.5)
    stats = []
    
    print(f"Analyzing {len(thresholds)} SNR thresholds...")
    
    for snr_cut in thresholds:
        filtered_df = results_df[results_df['power'] >= snr_cut].copy()
        n_candidates = len(filtered_df)
        
        # 完備性の計算
        recovered = 0
        results_dict = filtered_df.set_index('tic_id').to_dict('index')
        for _, toi in detectable_tois.iterrows():
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
        
        comp = recovered / len(detectable_tois)
        stats.append({
            'threshold': snr_cut,
            'candidates': n_candidates,
            'completeness': comp
        })
        
    stats_df = pd.DataFrame(stats)
    
    print("\n# Priority 4: SNR Threshold Sensitivity Analysis")
    print("| SNR Threshold | Candidates | Completeness | Candidates Reduction |")
    print("| :--- | :--- | :--- | :--- |")
    
    baseline_n = stats_df[stats_df['threshold'] == 7.0]['candidates'].iloc[0] if not stats_df[stats_df['threshold'] == 7.0].empty else stats_df.iloc[0]['candidates']
    
    for _, row in stats_df.iterrows():
        reduction = (1 - row['candidates'] / baseline_n) * 100
        print(f"| {row['threshold']:.1f} | {int(row['candidates']):,} | {row['completeness']:.2%} | {reduction:.1f}% |")

    # 最適値の推計
    # 完備性が15%以上を維持しつつ、候補数が最小になる点
    optimal = stats_df[stats_df['completeness'] >= 0.15].iloc[-1]
    print(f"\n✅ **Optimal Threshold Estimate**: **SNR > {optimal['threshold']:.1f}**")
    print(f"これにより完備性を維持しつつ、候補数を {int(optimal['candidates']):,} 件まで圧縮可能です。")

if __name__ == "__main__":
    run_snr_threshold_analysis()
