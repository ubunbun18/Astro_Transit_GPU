import matplotlib.pyplot as plt
import numpy as np
import os

def plot_folded_candidate(time, flux, period, t0, out_path, title=None):
    """
    Creates a folded light curve plot for a candidate.
    """
    # 1. Fold
    fold_time = (time - t0 + 0.5 * period) % period - 0.5 * period
    
    # Sort for cleaner plotting
    idx = np.argsort(fold_time)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(fold_time[idx], flux[idx], s=1, color='gray', alpha=0.3, label='Raw Data')
    
    # 2. Add a binned version for visibility
    try:
        from scipy.stats import binned_statistic
        # Dynamically adjust bins based on data density
        nbins = min(100, len(time) // 20)
        bin_means, bin_edges, _ = binned_statistic(fold_time, flux, statistic='mean', bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, bin_means, color='red', lw=2, label='Binned Mean', alpha=0.8)
        
        # Add a baseline line
        plt.axhline(0, color='white', lw=0.5, ls='--')
    except:
        pass

    plt.xlabel(f"Phase (days, P={period:.4f} d)")
    plt.ylabel("Relative Flux (Centered)")
    if title:
        plt.title(title)
    else:
        plt.title(f"Folded LC (P={period:.6f} d, T0={t0:.4f})")
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def generate_top_plots(candidates_df, sector_data, out_dir, top_n=20):
    """
    Generates plots for the top N candidates in a directory.
    Updates the dataframe with a 'plot_path' column.
    """
    top_n = int(top_n)
    os.makedirs(out_dir, exist_ok=True)
    
    # Check cache format
    is_vectorized = sector_data.get('is_vectorized', False)
    tic_ids = sector_data['tic_ids']
    id_to_idx = {int(tid): i for i, tid in enumerate(tic_ids)}
    
    # Initialize plot_path column
    df = candidates_df.copy()
    if 'plot_path' not in df.columns:
        df['plot_path'] = ""
        
    count = 0
    # Use vetting_score for ranking
    sort_key = 'vetting_score' if 'vetting_score' in df.columns else 'power'
    
    # We iterate over the sorted dataframe but update the original copy
    sorted_df = df.sort_values(sort_key, ascending=False)
    
    for idx, row in sorted_df.iterrows():
        if count >= top_n:
            break
            
        tid = int(row['tic_id'])
        if tid not in id_to_idx:
            continue
            
        rank = count + 1
        period = row['period']
        
        # New robust filename
        plot_name = f"TIC_{tid}_rank{rank}_P{period:.5f}_folded.png"
        out_path = os.path.join(out_dir, plot_name)
        
        idx_in_cache = id_to_idx[tid]
        
        if is_vectorized:
            time = sector_data['time']
            flux = sector_data['flux'][idx_in_cache]
            flux_err = sector_data['flux_err'][idx_in_cache]
            mask = flux_err < 0.99
            t_plot = time[mask]
            f_plot = flux[mask]
        else:
            offsets = sector_data['offsets']
            start = offsets[idx_in_cache]
            end = offsets[idx_in_cache+1] if idx_in_cache+1 < len(offsets) else len(sector_data['flux'])
            t_plot = sector_data['time'][start:end]
            f_plot = sector_data['flux'][start:end]
            
        if len(f_plot) == 0:
            continue
            
        title = f"TIC {tid} | Rank {rank} | Score: {row.get('vetting_score', 0):.2f} | P: {period:.4f} d"
        
        plot_folded_candidate(t_plot, f_plot, period, row['t0'], out_path, title=title)
        
        # Record relative path for HTML
        df.at[idx, 'plot_path'] = f"plots/{plot_name}"
        count += 1
        
    return df
