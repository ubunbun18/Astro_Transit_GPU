import argparse
import os
import sys
import time
import subprocess

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

def run_command(cmd, desc):
    print(f"\n>>> Starting: {desc}")
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            print(f"!!! Error during {desc} (Exit code: {process.returncode})")
            return False
    except Exception as e:
        print(f"!!! Exception during {desc}: {e}")
        return False
    
    elapsed = time.time() - start_time
    print(f"--- Completed: {desc} in {elapsed:.1f}s ---")
    return True

def main():
    parser = argparse.ArgumentParser(description="Full Pipeline: 1M Target Screening (QLP Style)")
    parser.add_argument("--sector", type=int, default=1, help="TESS Sector (default: 1)")
    parser.add_argument("--data-dir", type=str, default="data/qlp_bulk", help="Directory for FITS files")
    parser.add_argument("--cache-dir", type=str, default="data/cache", help="Directory for binary cache")
    parser.add_argument("--out", type=str, default="outputs", help="Directory for results")
    parser.add_argument("--workers-dl", type=int, default=100, help="Download workers")
    parser.add_argument("--workers-cache", type=int, default=12, help="Cache building workers (CPU)")
    parser.add_argument("--n-periods", type=int, default=5000, help="Number of test periods")
    parser.add_argument("--skip-dl", action="store_true", help="Skip download step")
    parser.add_argument("--skip-cache", action="store_true", help="Skip cache building step")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sector_str = f"s{args.sector:04d}"
    fits_dir = os.path.join(args.data_dir, sector_str)
    sector_cache_dir = os.path.join(args.cache_dir, sector_str)
    output_csv = os.path.join(args.out, f"screening_{sector_str}_final.csv")

    print(f"====================================================")
    print(f"   AstroTransit-GPU: 1M Target Pipeline ({sector_str})")
    print(f"====================================================\n")

    # 1. Download
    if not args.skip_dl:
        cmd_dl = [
            sys.executable, "scripts/download_ffi_bulk.py",
            "--sectors", str(args.sector),
            "--outdir", args.data_dir,
            "--workers", str(args.workers_dl)
        ]
        if not run_command(cmd_dl, "Bulk Download"):
            sys.exit(1)
    else:
        print(">>> Skipping Download Step")

    # 2. Build Cache
    if not args.skip_cache:
        cmd_cache = [
            sys.executable, "-m", "astrotransit_gpu", "build-cache",
            "--fits-dir", fits_dir,
            "--out-dir", sector_cache_dir,
            "--workers", str(args.workers_cache)
        ]
        if not run_command(cmd_cache, "Building Binary Cache"):
            sys.exit(1)
    else:
        print(">>> Skipping Cache Building Step")

    # 3. GPU Screening
    cmd_screen = [
        sys.executable, "-m", "astrotransit_gpu", "screen-sector",
        "--cache-dir", sector_cache_dir,
        "--out", output_csv,
        "--n-periods", str(args.n_periods),
        "--precision", "float32"
    ]
    if not run_command(cmd_screen, "GPU Screening (1M targets)"):
        sys.exit(1)

    print(f"\nSuccess! Final results saved to: {output_csv}")

if __name__ == "__main__":
    main()
