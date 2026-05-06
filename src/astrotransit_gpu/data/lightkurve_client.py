import lightkurve as lk
import os
import logging

logger = logging.getLogger(__name__)

class LightkurveClient:
    """Wrapper for downloading light curves from MAST using Lightkurve."""
    
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def download_lightcurve(self, target_id, mission="TESS", author="SPOC", sector=None, quarter=None):
        """
        Download a light curve for a given target.
        
        Args:
            target_id (str): Target identifier (e.g., 'TIC 261136679' or 'KIC 8462852').
            mission (str): 'TESS' or 'Kepler'.
            author (str): Data product author (e.g., 'SPOC', 'QLP', 'Kepler').
            sector (int): Sector number for TESS.
            quarter (int): Quarter number for Kepler.
            
        Returns:
            lk.LightCurve: The downloaded light curve.
        """
        logger.info(f"Searching for {target_id} ({mission}, {author})...")
        
        search_result = lk.search_lightcurve(
            target_id, 
            mission=mission, 
            author=author,
            sector=sector,
            quarter=quarter
        )
        
        if len(search_result) == 0:
            raise ValueError(f"No light curves found for {target_id}")
            
        logger.info(f"Found {len(search_result)} datasets. Downloading the first one...")
        lc = search_result[0].download(download_dir=self.cache_dir)
        
        if lc is None:
            raise RuntimeError(f"Failed to download light curve for {target_id}")
            
        return lc
