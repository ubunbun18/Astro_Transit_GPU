import pandas as pd
import os

class CatalogMatch:
    """Handles matching candidates with known TOI/EB catalogs."""
    
    def __init__(self, toi_path=None, eb_path=None):
        self.toi_ids = set()
        self.eb_ids = set()
        
        if toi_path and os.path.exists(toi_path):
            try:
                toi_df = pd.read_csv(toi_path)
                # Flexible column detection
                possible_cols = ['tic_id', 'TIC', 'tid', 'TICID', 'TIC ID']
                col = next((c for c in possible_cols if c in toi_df.columns), None)
                if col:
                    self.toi_ids = set(toi_df[col].dropna().astype(int).tolist())
            except Exception as e:
                print(f"Warning: Failed to load TOI catalog: {e}")
                
        if eb_path and os.path.exists(eb_path):
            try:
                eb_df = pd.read_csv(eb_path)
                possible_cols = ['tic_id', 'TIC', 'tid', 'TICID', 'TIC ID']
                col = next((c for c in possible_cols if c in eb_df.columns), None)
                if col:
                    self.eb_ids = set(eb_df[col].dropna().astype(int).tolist())
            except Exception as e:
                print(f"Warning: Failed to load EB catalog: {e}")
                
    def match(self, tic_id):
        """Returns the type of the target if matched."""
        tid = int(tic_id)
        if tid in self.toi_ids:
            return "TOI"
        if tid in self.eb_ids:
            return "EB"
        return "unknown"

def apply_catalog_matching(df, toi_path=None, eb_path=None):
    """Augments dataframe with known_type column."""
    matcher = CatalogMatch(toi_path, eb_path)
    df = df.copy()
    df['known_type'] = df['tic_id'].apply(matcher.match)
    return df
