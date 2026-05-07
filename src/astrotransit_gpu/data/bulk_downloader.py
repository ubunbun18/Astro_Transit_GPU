import os
import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm
from astroquery.mast import Observations
import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

class BulkDownloader:
    """
    High-performance async downloader for TESS light curves using MAST bulk products.
    Optimized for downloading hundreds of thousands of targets using asyncio and aiohttp.
    """
    
    def __init__(self, base_dir: str, workers: int = 50):
        self.base_dir = base_dir
        self.workers = workers
        self.semaphore = asyncio.Semaphore(workers)
        os.makedirs(base_dir, exist_ok=True)

    async def download_sector_products(self, sector: int, product_type: str = "QLP"):
        """
        Specified in bulk_downloader_spec.md: 
        Query and download all products for a specific sector.
        """
        manifest = await self.get_sector_manifest(sector, product_type)
        if manifest.empty:
            print(f"No products found for Sector {sector} {product_type}")
            return {"ok": 0, "skipped": 0, "error": 0}

        sub_dir = f"s{sector:04d}"
        target_dir = os.path.join(self.base_dir, sub_dir)
        os.makedirs(target_dir, exist_ok=True)

        tasks = []
        for _, row in manifest.iterrows():
            url = row['url']
            if url.startswith("mast:"):
                url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={url}"
            filename = row['filename']
            save_path = os.path.join(target_dir, filename)
            tasks.append((url, save_path))

        print(f"Starting async download of {len(tasks)} files for Sector {sector} (Workers: {self.workers})")
        return await self.download_from_list(tasks)

    async def get_sector_manifest(self, sector: int, product_type: str = "QLP") -> pd.DataFrame:
        """
        Query MAST for all light curve products in a specific sector.
        Tries fast manifest first, then falls back to Observations query.
        """
        if product_type == "QLP":
            df = await self._get_sector_manifest_fast(sector)
            if not df.empty:
                return df
            
        print(f"Querying MAST via astroquery for Sector {sector} {product_type} light curves...")
        # astroquery is synchronous, we run it in a thread to avoid blocking the loop
        loop = asyncio.get_event_loop()
        try:
            def sync_query():
                obs = Observations.query_criteria(
                    obs_collection='TESS',
                    project='TESS',
                    provenance_name=product_type,
                    sequence_number=sector
                )
                if len(obs) == 0:
                    return pd.DataFrame()
                products = Observations.get_product_list(obs)
                filtered = Observations.filter_products(
                    products, 
                    productSubGroupDescription="LIGHTCURVE",
                    extension="fits"
                )
                res_df = filtered.to_pandas()
                res_df = res_df.rename(columns={"dataURI": "url"})
                res_df['filename'] = res_df['url'].apply(lambda x: os.path.basename(x))
                return res_df

            return await loop.run_in_executor(None, sync_query)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return pd.DataFrame()

    async def _get_sector_manifest_fast(self, sector: int) -> pd.DataFrame:
        """
        Fetch the QLP manifest by parsing the official MAST curl script.
        Uses requests for the large manifest file and then proceeds with async.
        """
        print(f"Fetching Fast Manifest for Sector {sector} QLP...")
        import requests
        
        versions = ["v01", "v02"]
        for v in versions:
            url = f"https://archive.stsci.edu/hlsps/qlp/download_scripts/hlsp_qlp_tess_ffi_s{sector:04d}_tess_{v}_llc-fits.sh"
            try:
                # Synchronous request for the manifest index (can be ~120MB)
                response = await asyncio.get_event_loop().run_in_executor(None, lambda: requests.get(url, timeout=300))
                if response.status_code == 200:
                    content = response.text
                    # Updated regex for 'curl -f --create-dirs --output 'path' 'url''
                    pattern = re.compile(r"--output '(\S+)' '(\S+)'")
                    matches = pattern.findall(content)
                    if matches:
                        df = pd.DataFrame(matches, columns=["path", "url"])
                        df['filename'] = df['path'].apply(lambda x: os.path.basename(x))
                        print(f"Found {len(df)} products via {v} script.")
                        return df[['filename', 'url']]
            except Exception as e:
                logger.debug(f"Fast manifest attempt {v} errored: {e}")
                continue
                
        return pd.DataFrame()

    async def download_from_list(self, tasks: List[Tuple[str, str]]):
        """
        Specified in bulk_downloader_spec.md:
        Execute parallel downloads for a list of (url, save_path) tuples.
        """
        if not tasks:
            return {"ok": 0, "skipped": 0, "error": 0}

        results = {"ok": 0, "skipped": 0, "error": 0}
        
        async with aiohttp.ClientSession() as session:
            # We use tqdm.asyncio to track progress of our coroutines
            pbar = tqdm(total=len(tasks), desc="Downloading")
            
            async def download_one(url, path):
                nonlocal results
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    results["skipped"] += 1
                    pbar.update(1)
                    return

                async with self.semaphore:
                    try:
                        retry_count = 0
                        max_retries = 3
                        while retry_count < max_retries:
                            try:
                                async with session.get(url, timeout=60) as response:
                                    if response.status == 200:
                                        temp_path = path + ".part"
                                        with open(temp_path, 'wb') as f:
                                            while True:
                                                chunk = await response.content.read(8192)
                                                if not chunk:
                                                    break
                                                f.write(chunk)
                                        os.rename(temp_path, path)
                                        results["ok"] += 1
                                        return
                                    elif response.status == 429:
                                        retry_count += 1
                                        wait_time = 2 ** retry_count
                                        logger.warning(f"Rate limited (429) for {url}. Waiting {wait_time}s... (Attempt {retry_count}/{max_retries})")
                                        await asyncio.sleep(wait_time)
                                        continue
                                    else:
                                        results["error"] += 1
                                        logger.error(f"Failed to download {url}: Status {response.status}")
                                        return
                            except Exception as e:
                                if retry_count < max_retries - 1:
                                    retry_count += 1
                                    await asyncio.sleep(1)
                                    continue
                                results["error"] += 1
                                logger.error(f"Error downloading {url}: {e}")
                                if os.path.exists(path + ".part"):
                                    try:
                                        os.remove(path + ".part")
                                    except: pass
                                return
                    finally:
                        pbar.update(1)

            await asyncio.gather(*(download_one(url, path) for url, path in tasks))
            pbar.close()

        print(f"\nSummary: OK={results['ok']}, Skipped={results['skipped']}, Error={results['error']}")
        return results
