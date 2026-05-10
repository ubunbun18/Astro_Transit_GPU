import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from .api import BoxLeastSquaresGPU
from ..data.sector_cache import SectorCache

class CandidateRefiner:
    """
    Refines screening results by performing detailed Top-K search on high-SNR targets.
    """
    
    def __init__(self, cache_dir, n_periods=10000):
        self.cache = SectorCache(cache_dir)
        self.n_periods = n_periods
        self._loaded_cache = None

    def refine_from_csv(self, input_csv, output_csv, config=None, toi_path=None, eb_path=None, top_k=5):
        """
        Reads results CSV, selects top targets based on multiple rules, 
        and runs detailed Top-K BLS.
        """
        df = pd.read_csv(input_csv)
        
        if config is None:
            config = {
                'snr_threshold': 7.1,
                'top_n_targets': 100,
                'random_sample_n': 5
            }
            
        print(f"Selecting targets for refinement from {len(df)} candidates...")

        # --- SELECTION RULES ---
        selected_indices = set()
        
        # Rule 1: SNR Threshold
        snr_thresh = float(config.get('snr_threshold', 7.1))
        r1 = df[df['power'] >= snr_thresh].index
        selected_indices.update(r1.tolist())
        
        # Rule 2: Top N targets by SNR
        top_n = int(config.get('top_n_targets', 0))
        if top_n > 0:
            r2 = df.sort_values('power', ascending=False).head(top_n).index
            selected_indices.update(r2.tolist())
            
        # Rule 3: Known TOI / EB
        if toi_path or eb_path:
            from ..vet.catalog import CatalogMatch
            matcher = CatalogMatch(toi_path, eb_path)
            # Find indices where tic_id is in catalogs (match returns 'toi' or 'eb')
            r3 = df[df['tic_id'].apply(lambda x: matcher.match(x) in ['toi', 'eb'])].index
            selected_indices.update(r3.tolist())
            
        # Rule 4 & 5: Heuristics
        p_cfg = config.get('planet_like', {})
        if p_cfg:
            min_p = float(p_cfg.get('min_power', 1.0e9))
            max_d = float(p_cfg.get('max_depth', 0.01))
            max_dur_f = float(p_cfg.get('max_duration_frac', 0.1))
            r5 = df[
                (df['power'] >= min_p) &
                (df['depth'] <= max_d) &
                (df['duration'] <= df['period'] * max_dur_f)
            ].index
            selected_indices.update(r5.tolist())
            
        a_cfg = config.get('artifact_like', {})
        if a_cfg:
            # EB / Artifact suspects
            r4 = df[
                (df['depth'] >= a_cfg.get('min_depth', 0.05)) |
                (df['duration'] >= df['period'] * a_cfg.get('min_duration_frac', 0.2))
            ].index
            selected_indices.update(r4.tolist())
            
        # Rule 6: Random Sample
        rand_n = config.get('random_sample_n', 0)
        if rand_n > 0 and len(df) > 0:
            # Sample from candidates not already selected
            remaining = df.drop(list(selected_indices))
            if not remaining.empty:
                r6 = remaining.sample(min(rand_n, len(remaining))).index
                selected_indices.update(r6.tolist())
                
        # Final target list
        targets_df = df.loc[list(selected_indices)]
        tic_ids = targets_df['tic_id'].unique().tolist()
        
        print(f"Refining {len(tic_ids)} unique targets based on multiple rules...")
        
        if not tic_ids:
            print("No targets found above threshold.")
            return None

        # 2. Load Cache
        self._loaded_cache = self.cache.load()
        
        all_candidates = []
        
        # 3. Batch Process
        for tid in tqdm(tic_ids, desc="Refining"):
            data = self.cache.get_target_data(tid, loaded_data=self._loaded_cache)
            if data is None:
                continue
                
            t, y, dy = data['time'], data['flux'], data['flux_err']
            
            # Use detailed period grid for refinement
            # Using the same range but more points
            p_min, p_max = 0.5, 20.0 # Default range
            periods = np.linspace(p_min, p_max, self.n_periods)
            durations = np.linspace(0.01, 0.2, 5)
            
            try:
                model = BoxLeastSquaresGPU(t, y, dy=dy)
                res = model.power(periods, durations)
                
                # Extract top K candidates
                # api.py already does this in model.power() return object
                for c in res.top_candidates[:top_k]:
                    all_candidates.append({
                        "tic_id": int(tid),
                        "status": "ok",
                        "period": float(c.period),
                        "t0": float(c.t0),
                        "depth": float(c.depth),
                        "duration": float(c.duration),
                        "power": float(c.power),
                        "n_points": int(len(t)),
                        "error": ""
                    })
            except Exception as e:
                print(f"Error refining TIC {tid}: {e}")
                
        # 4. Save results
        if all_candidates:
            refined_df = pd.DataFrame(all_candidates)
            refined_df.to_csv(output_csv, index=False)
            print(f"Refinement complete. Saved {len(refined_df)} candidates to {output_csv}")
            return refined_df
        else:
            print("No candidates found during refinement.")
            return None
