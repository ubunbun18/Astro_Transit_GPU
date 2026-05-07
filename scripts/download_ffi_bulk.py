import argparse
import sys
import os
import asyncio

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    from astrotransit_gpu.data.bulk_downloader import BulkDownloader
except ImportError:
    print("Error: Could not import BulkDownloader. Make sure dependencies are installed.")
    print("Please run: pip install astroquery aiohttp tqdm pandas")
    sys.exit(1)

async def main():
    parser = argparse.ArgumentParser(description="Download ~1M TESS FFI targets in bulk (Async).")
    parser.add_argument("--sectors", type=str, default="1-7", help="Sector range (e.g., '1-7' or '1,2,3')")
    parser.add_argument("--outdir", type=str, default="data/tess_ffi_bulk", help="Output directory")
    parser.add_argument("--type", type=str, default="QLP", choices=["QLP", "SPOC", "TESS-SPOC"], help="Product type")
    parser.add_argument("--workers", type=int, default=50, help="Parallel download workers (async semaphore)")
    args = parser.parse_args()

    # Parse sectors
    sector_list = []
    if "-" in args.sectors:
        start, end = map(int, args.sectors.split("-"))
        sector_list = list(range(start, end + 1))
    else:
        sector_list = [int(s) for s in args.sectors.split(",")]

    downloader = BulkDownloader(base_dir=args.outdir, workers=args.workers)

    print(f"=== TESS Bulk Downloader (Async Mode) ===")
    print(f"Target Sectors: {sector_list}")
    print(f"Product Type:   {args.type}")
    print(f"Output Dir:     {args.outdir}")
    print(f"Workers:        {args.workers} (Semaphore)")
    print(f"=========================================\n")

    for sector in sector_list:
        print(f"\n--- Processing Sector {sector} ---")
        # Use the combined high-level async method
        await downloader.download_sector_products(sector, product_type=args.type)

    print("\nAll tasks completed.")

if __name__ == "__main__":
    asyncio.run(main())
