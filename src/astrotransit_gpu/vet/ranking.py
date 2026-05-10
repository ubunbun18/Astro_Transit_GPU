import numpy as np

def calculate_vetting_scores(df, config=None):
    """
    Calculates a vetting score for each candidate.
    """
    if config is None:
        config = {}
        
    cfg = config.get('scoring', {})
    snr_norm = float(cfg.get('snr_norm', 1e9))
    snr_weight = float(cfg.get('snr_weight', 1.0))
    plaus_weight = float(cfg.get('plausibility_weight', 1.0))
    
    df = df.copy()
    
    # 1. Distinguish raw power and SNR
    if 'raw_power' not in df.columns:
        df['raw_power'] = df['power']
    
    # Normalized SNR
    df['snr'] = df['raw_power'] / snr_norm
    
    # 2. SNR Score (using log scale to handle high dynamic range)
    # 0.1 at snr=0, 1.0 at snr=100
    df['snr_score'] = np.clip(np.log10(df['snr'] + 1) / 2.0, 0, 1)
    
    # 3. Plausibility Score
    df['plausibility_score'] = 1.0
    
    # Hard physical limit: duration < period/2
    mask_impossible = df['duration'] > (df['period'] * 0.5)
    df.loc[mask_impossible, 'plausibility_score'] *= 0.1
    
    # Wide transits are suspicious for planets
    mask_wide = df['duration'] > (df['period'] * 0.2)
    df.loc[mask_wide, 'plausibility_score'] *= 0.5
    
    # Very deep transits are likely EBs
    mask_deep = df['depth'] > 0.05
    df.loc[mask_deep, 'plausibility_score'] *= 0.7

    # 4. Final Score calculation
    base_score = (df['snr_score'] * snr_weight + df['plausibility_score'] * plaus_weight) / (snr_weight + plaus_weight)
    
    # 5. Catalog Adjustments (Lowercase)
    if 'known_type' in df.columns:
        base_score[df['known_type'] == 'eb'] += cfg.get('known_eb_penalty', -0.8)
        base_score[df['known_type'] == 'toi'] += cfg.get('known_toi_bonus', 0.5)
        
    df['vetting_score'] = np.clip(base_score, 0, 1)
    
    return df.sort_values('vetting_score', ascending=False)
