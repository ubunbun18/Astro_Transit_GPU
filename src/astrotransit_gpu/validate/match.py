def match_candidate(p_detected, t0_detected, p_true, t0_true, p_tol=0.01, t0_tol=0.1):
    """
    Match a detected candidate with a true planet period and epoch.
    """
    if p_detected <= 0 or p_true <= 0:
        return {'is_match': False, 'match_type': 'invalid_input'}

    # 1. Direct match
    diff = abs(p_detected - p_true) / p_true
    if diff < p_tol:
        return {'is_match': True, 'match_type': 'direct', 'p_diff': diff}
    
    # 2. Harmonics (P_det = n * P_true)
    for n in [2, 3]:
        diff = abs(p_detected - n * p_true) / (n * p_true)
        if diff < p_tol:
            return {'is_match': True, 'match_type': 'harmonic', 'p_diff': diff}
            
    # 3. Sub-harmonics (P_det = P_true / n)
    for n in [2, 3]:
        diff = abs(p_detected - p_true / n) / (p_true / n)
        if diff < p_tol:
            return {'is_match': True, 'match_type': 'subharmonic', 'p_diff': diff}
            
    return {'is_match': False, 'match_type': 'none', 'p_diff': abs(p_detected - p_true) / p_true}
