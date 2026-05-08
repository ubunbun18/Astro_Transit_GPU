import pandas as pd
from astrotransit_gpu.data.exoplanet_archive import ExoplanetArchiveClient

def debug_validation():
    results_df = pd.read_csv("outputs/bench_219331_v39.csv")
    results_df['tic_id'] = results_df['tic_id'].astype(str)
    
    toi_df = ExoplanetArchiveClient.get_toi_table()
    toi_df['tid'] = toi_df['tid'].astype(str)
    
    sample_tic_ids = set(results_df['tic_id'])
    tois_in_sample = toi_df[toi_df['tid'].isin(sample_tic_ids)].copy()
    
    results_dict = results_df.set_index('tic_id').to_dict('index')
    
    print(f"TIC ID | Det P | True P | Det SNR | Det Dep | True Dep")
    print("-" * 65)
    
    count = 0
    for _, toi in tois_in_sample.iterrows():
        tic_id = toi['tid']
        det = results_dict[tic_id]
        p_det = det['period']
        p_true = toi['pl_orbper']
        
        # Check for harmonics
        is_close = False
        for factor in [0.5, 1.0, 2.0]:
            if abs(p_det - p_true * factor) / (p_true * factor) < 0.05:
                is_close = True
                break

        if is_close or det['power'] > 5:
            true_dep = toi['pl_trandep'] / 1e6 # ppm to relative
            print(f"{tic_id} | {p_det:.3f} | {p_true:.3f} | {det['power']:.2f} | {det['depth']:.4f} | {true_dep:.4f}")
            count += 1
        if count > 30: break

    max_p = 0
    best_tic = ""
    for _, toi in tois_in_sample.iterrows():
        tic_id = toi['tid']
        det = results_dict[tic_id]
        if det['power'] > max_p:
            max_p = det['power']
            best_tic = tic_id
            
    print(f"\nMax Power among TOIs: {max_p:.2f} (TIC {best_tic})")
    
    # Check if there are any infs in TOIs
    inf_tois = tois_in_sample[tois_in_sample['tid'].apply(lambda x: results_dict[x]['power'] == float('inf'))]
    print(f"Number of TOIs with inf power: {len(inf_tois)}")

if __name__ == "__main__":
    debug_validation()
