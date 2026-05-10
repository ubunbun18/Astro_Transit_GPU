import pandas as pd
import numpy as np

def group_harmonics(df, tolerance=0.01):
    """
    Groups candidates by tic_id and period harmonics.
    
    Args:
        df (pd.DataFrame): Dataframe with ['tic_id', 'period']
        tolerance (float): Relative tolerance for harmonic detection.
        
    Returns:
        pd.DataFrame: Augmented dataframe with 'harmonic_group_id', 'harmonic_relation', 'canonical_period'
    """
    if df.empty:
        return df
        
    df = df.copy()
    df['harmonic_group_id'] = -1
    df['harmonic_relation'] = ""
    df['canonical_period'] = 0.0
    
    group_counter = 0
    
    # Common harmonic ratios (n/m)
    ratios = [
        (1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1)
    ]
    
    for tic_id, group in df.groupby('tic_id'):
        indices = group.index.tolist()
        processed = set()
        
        # Sort by period to find canonical (usually shortest) period first
        sorted_indices = group.sort_values('period').index.tolist()
        
        for idx in sorted_indices:
            if idx in processed:
                continue
                
            canonical_p = df.at[idx, 'period']
            df.at[idx, 'harmonic_group_id'] = group_counter
            df.at[idx, 'harmonic_relation'] = "1:1"
            df.at[idx, 'canonical_period'] = canonical_p
            processed.add(idx)
            
            # Look for harmonics in remaining unprocessed candidates of the same TIC
            for other_idx in sorted_indices:
                if other_idx in processed:
                    continue
                    
                other_p = df.at[other_idx, 'period']
                
                found_relation = False
                for n, m in ratios:
                    target_p = canonical_p * (n / m)
                    if abs(other_p - target_p) / target_p < tolerance:
                        df.at[other_idx, 'harmonic_group_id'] = group_counter
                        df.at[other_idx, 'harmonic_relation'] = f"{n}:{m}"
                        df.at[other_idx, 'canonical_period'] = canonical_p
                        processed.add(other_idx)
                        found_relation = True
                        break
            
            group_counter += 1
            
    df['is_harmonic'] = df['harmonic_relation'].apply(lambda x: x != "1:1" and x != "")
    return df
