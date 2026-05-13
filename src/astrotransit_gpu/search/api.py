from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Dict, Any, Union
from .gpu_bls import run_gpu_bls, get_top_k_candidates
from .v42_parity import run_vbls_exact_parity

@dataclass
class Candidate:
    period: float
    t0: float
    duration: float
    depth: float
    power: float

@dataclass
class BLSResult:
    best_period: float
    best_t0: float
    best_duration: float
    best_depth: float
    best_power: float
    periods: np.ndarray
    power: np.ndarray
    top_candidates: List[Candidate]
    metadata: Dict[str, Any]

class BoxLeastSquaresGPU:
    """
    Astropy-compatible GPU-accelerated Box Least Squares (BLS).
    """
    def __init__(self, t: np.ndarray, y: np.ndarray, dy: Optional[np.ndarray] = None):
        self.t = t
        self.y = y
        self.dy = dy
        
    def power(self, periods: np.ndarray, durations: np.ndarray, 
              n_bins: int = 200, dtype: Any = np.float32,
              method: str = "fast", **kwargs) -> BLSResult:
        """
        Compute the BLS power spectrum.
        
        Args:
            periods: Array of periods to search.
            durations: Array of transit durations to search.
            n_bins: Number of bins for phase folding (Fast mode only).
            dtype: Data type for computation.
            method: 'fast' (V41/V39) or 'parity' (V42, Astropy compatible).
            **kwargs: Additional parameters (e.g., max_bins for parity mode).
        """
        # 1. Input Validation
        periods = np.atleast_1d(periods)
        durations = np.atleast_1d(durations)
        
        if np.any(periods <= 0):
            raise ValueError("All periods must be positive.")
        if np.any(durations <= 0):
            raise ValueError("All durations must be positive.")
        if np.any(durations >= np.min(periods)):
            # Warning or Error? Astropy allows it but it's physically weird. 
            # We'll allow it but ensure n_bins is enough.
            pass
            
        if np.any(np.isnan(self.t)) or np.any(np.isnan(self.y)):
            raise ValueError("Input time or flux contains NaNs.")
            
        if self.dy is not None:
            if np.any(self.dy <= 0):
                raise ValueError("flux_err must be strictly positive for weighted BLS.")
            if len(self.dy) != len(self.y):
                raise ValueError("flux_err and flux must have the same length.")
        
        # Run core GPU search
        if method == "parity":
            # V42 Parity Mode
            max_bins = kwargs.get("max_bins", 2000)
            oversample = kwargs.get("oversample", 10)
            
            # Reshape flux for multi-target kernel (even if single target)
            flux_2d = self.y.reshape(1, -1)
            weights_2d = None
            if self.dy is not None:
                weights_2d = (1.0 / (self.dy**2)).reshape(1, -1)
            
            raw_res_v42 = run_vbls_exact_parity(
                self.t, flux_2d, periods, durations, 
                weights_matrix=weights_2d, oversample=oversample, 
                max_bins=max_bins, dtype=dtype
            )
            
            # Convert V42 output to expected format
            raw_res = {
                "best_period": float(raw_res_v42['best_period'][0]),
                "best_t0": float(raw_res_v42['best_t0'][0]),
                "best_duration": float(raw_res_v42['best_duration'][0]),
                "best_depth": float(raw_res_v42['best_depth'][0]),
                "snr": float(raw_res_v42['snr'][0]),
                "power": raw_res_v42['power_array'][0],
                "all_t0s": raw_res_v42['t0_array'][0],
                "all_durs": raw_res_v42['dur_array'][0],
                "all_depths": raw_res_v42['depth_array'][0],
                "periods": periods
            }
            n_bins_used = max_bins # metadata
        else:
            # V41 Fast Mode
            raw_res = run_gpu_bls(
                self.t, self.y, periods, durations, 
                flux_err=self.dy, n_bins=n_bins, dtype=dtype
            )
            n_bins_used = n_bins
        
        # Extract Top-K candidates
        top_k_raw = get_top_k_candidates(raw_res, k=5)
        top_candidates = [
            Candidate(
                float(c['period']), 
                float(c['t0']), 
                float(c['duration']), 
                float(c['depth']), 
                float(c['power'])
            )
            for c in top_k_raw
        ]
        
        return BLSResult(
            best_period=float(raw_res['best_period']),
            best_t0=float(raw_res['best_t0']),
            best_duration=float(raw_res['best_duration']),
            best_depth=float(raw_res['best_depth']),
            best_power=float(raw_res['snr']),
            periods=periods,
            power=raw_res['power'].get() if hasattr(raw_res['power'], 'get') else raw_res['power'],
            top_candidates=top_candidates,
            metadata={
                "n_bins": n_bins_used,
                "dtype": str(dtype),
                "n_data": len(self.t),
                "method": method
            }
        )
