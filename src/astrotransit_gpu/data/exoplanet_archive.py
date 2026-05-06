import pandas as pd
import requests
import io
import logging

logger = logging.getLogger(__name__)

class ExoplanetArchiveClient:
    """Client for fetching known planet data from NASA Exoplanet Archive."""
    
    TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    @classmethod
    def get_toi_table(cls):
        """Fetch the TESS Objects of Interest (TOI) table."""
        query = "select * from toi"
        params = {
            "query": query,
            "format": "csv"
        }
        logger.info("Fetching TOI table from NASA Exoplanet Archive...")
        response = requests.get(cls.TAP_URL, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch TOI table: {response.status_code}")
            
        return pd.read_csv(io.StringIO(response.text))

    @classmethod
    def get_confirmed_planets(cls):
        """Fetch the confirmed planets table (pscomppars)."""
        query = "select * from pscomppars where default_flag = 1"
        params = {
            "query": query,
            "format": "csv"
        }
        logger.info("Fetching confirmed planets table...")
        response = requests.get(cls.TAP_URL, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch confirmed planets: {response.status_code}")
            
        return pd.read_csv(io.StringIO(response.text))

    @classmethod
    def get_toi_targets(cls, n_targets=10, min_depth_ppm=500):
        """Get a list of TOI targets with certain criteria."""
        df = cls.get_toi_table()
        # Filter for confirmed or high-quality candidates
        # Correct column names based on TAP 'toi' table schema
        mask = (df['pl_trandep'] >= min_depth_ppm) & (df['tfopwg_disp'].isin(['PC', 'KP']))
        df_filtered = df[mask].head(n_targets)
        
        targets = []
        for _, row in df_filtered.iterrows():
            targets.append({
                'target_id': f"TIC {row['tid']}",
                'period': row['pl_orbper'],
                't0': row['pl_tranmid'],
                'depth': row['pl_trandep'] / 1e6, # ppm to relative
                'duration': row['pl_trandurh'] / 24.0 # hours to days
            })
        return targets
