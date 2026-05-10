import numpy as np

def calculate_vetting_scores(df, config=None):
    """
    Calculates a vetting score for each candidate.
    
    Args:
        df (pd.DataFrame): Candidates with ['power', 'period', 'depth', 'duration']
        config (dict): Scoring weights and thresholds.
        
    Returns:
        pd.DataFrame: Augmented with 'vetting_score'
    """
    if config is None:
        config = {
            'snr_weight': 1.0,
            'snr_norm': 1e9, # Typical large scale from current GPU kernel
            'plausibility_weight': 1.0,
            'known_toi_bonus': 0.5,
            'known_eb_penalty': -0.8
        }
        
    df = df.copy()
    
    # 1. SNR Score (Normalized)
    # Using 'power' column which contains SNR or raw power from GPU
    snr_norm = config.get('snr_norm', 1e9)
    df['snr_score'] = np.clip(df['power'] / snr_norm, 0, 1)
    
    # 2. Plausibility Score
    # Physical constraints: duration < period/2
    # Also very short durations relative to period might be artifacts
    df['plausibility_score'] = 1.0
    
    # Hard physical limit
    mask_impossible = df['duration'] > (df['period'] * 0.5)
    df.loc[mask_impossible, 'plausibility_score'] *= 0.1
    
    # Typical transit duration is small fraction of period
    # If duration > 20% of period, likely EB or artifact
    mask_wide = df['duration'] > (df['period'] * 0.2)
    df.loc[mask_wide, 'plausibility_score'] *= 0.5
    
    # 3. Catalog Scores (if column exists)
    df['catalog_modifier'] = 0.0
    if 'known_type' in df.columns:
        df.loc[df['known_type'] == 'EB', 'catalog_modifier'] = config.get('known_eb_penalty', -0.8)
        df.loc[df['known_type'] == 'TOI', 'catalog_modifier'] = config.get('known_toi_bonus', 0.5)
        
    # 4. Final Score
    # Score = (SNR * Plausibility) + CatalogModifier
    df['vetting_score'] = (df['snr_score'] * config.get('snr_weight', 1.0) * df['plausibility_score']) + df['catalog_modifier']
    
    # Clip to [0, 1]
    df['vetting_score'] = np.clip(df['vetting_score'], 0, 1)
    
    return df
