import asyncio
from astrotransit_gpu.data.bulk_downloader import BulkDownloader
import os

async def download_sector_2_limited(limit=5000):
    sector = 2
    downloader = BulkDownloader(base_dir="data/tess_data")
    
    # マニフェストの取得
    manifest = await downloader.get_sector_manifest(sector, product_type="QLP")
    if manifest.empty:
        print("Manifest not found.")
        return
    
    # 先頭5000件に制限
    limited_manifest = manifest.iloc[:limit]
    print(f"Limited manifest to {len(limited_manifest)} products.")
    
    sub_dir = f"s{sector:04d}"
    target_dir = os.path.join(downloader.base_dir, sub_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    tasks = []
    for _, row in limited_manifest.iterrows():
        url = row['url']
        if url.startswith("mast:"):
            url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={url}"
        filename = row['filename']
        save_path = os.path.join(target_dir, filename)
        tasks.append((url, save_path))
        
    print(f"Starting limited download of Sector {sector}...")
    await downloader.download_from_list(tasks)

if __name__ == "__main__":
    asyncio.run(download_sector_2_limited(15000))
