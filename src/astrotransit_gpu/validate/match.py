def match_candidate(p_detected, t0_detected, p_true, t0_true, p_tol=0.01, t0_tol=0.1):
    """
    Check if a detected candidate matches a true signal with phase consistency.
    """
    if p_true <= 0 or p_detected <= 0:
        return {"is_match": False, "match_type": "none"}

    ratios = {
        "direct": 1.0,
        "harmonic": 2.0,
        "harmonic_3": 3.0,
        "subharmonic": 0.5,
        "subharmonic_3": 1/3
    }
    
    for m_type, ratio in ratios.items():
        p_target = p_true * ratio
        p_diff = abs(p_detected - p_target) / p_target
        
        if p_diff < p_tol:
            # Phase check (T0 matching)
            # Use detected period for phase wrapping
            phase_diff = abs((t0_detected - t0_true) % p_detected)
            phase_diff = min(phase_diff, p_detected - phase_diff)
            
            if phase_diff < t0_tol:
                return {
                    "is_match": True, 
                    "match_type": m_type, 
                    "p_diff": p_diff,
                    "t0_diff": phase_diff
                }
            
    return {"is_match": False, "match_type": "none", "p_diff": abs(p_detected - p_true) / p_true}
