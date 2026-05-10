import argparse
import sys
import numpy as np
import time
import yaml
import pandas as pd
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from .data.lightkurve_client import LightkurveClient
from .preprocess.clean import clean_lightcurve, to_arrays
from .search.api import BoxLeastSquaresGPU
from .search.cpu_reference_bls import run_astropy_bls
from .inject.grid import run_injection_recovery_experiment, calculate_recovery_map
from .data.sector_cache import SectorCache
import lightkurve as lk

def download_and_preprocess(target_id, data_dir=None, max_retries=2):
    """
    Download (if needed) and preprocess a light curve.
    Checks data_dir for existing files to avoid redundant downloads.
    """
    t_start = time.time()
    
    for attempt in range(max_retries):
        try:
            lc = None
            t_downloaded = t_start
            
            # 1. Check for existing file in flat data_dir (from bulk download)
            target_clean = str(target_id).replace("TIC", "").strip()
            if data_dir:
                import glob
                padded_id = target_clean.zfill(16)
                matches = glob.glob(os.path.join(data_dir, f"*{padded_id}*.fits"))
                if matches:
                    try:
                        lc = lk.read(matches[0])
                        t_downloaded = time.time()
                    except Exception as e:
                        # If file is corrupt, we'll try to download it via Lightkurve
                        pass

            # 2. Download via Lightkurve if not found or corrupted
            if lc is None:
                client = LightkurveClient(cache_dir=data_dir)
                lc = client.download_lightcurve(f"TIC {target_clean}")
                t_downloaded = time.time()
            
            # 3. Preprocess
            lc_clean = clean_lightcurve(lc)
            t, y = to_arrays(lc_clean)
            dy = lc_clean.flux_err.value if hasattr(lc_clean, 'flux_err') else None
            t_preprocessed = time.time()
            
            return {
                "target_id": target_id,
                "status": "ok",
                "time_array": t,
                "flux_array": y,
                "dy_array": dy,
                "download_time": t_downloaded - t_start,
                "preprocess_time": t_preprocessed - t_downloaded,
                "n_data": len(t),
                "error": ""
            }

        except Exception as e:
            err_msg = str(e)
            is_corrupt = any(k in err_msg.lower() for k in ["corrupt", "truncated", "end of file"])
            
            if is_corrupt and attempt < max_retries - 1:
                try:
                    import shutil
                    # Cleanup structured cache if it exists
                    cache_base = os.path.expanduser("~/.lightkurve/cache/mastDownload/TESS")
                    if os.path.exists(cache_base):
                        target_clean = str(target_id).replace("TIC", "").strip().zfill(16)
                        for root, dirs, files in os.walk(cache_base):
                            if target_clean in root:
                                shutil.rmtree(root)
                                break
                except:
                    pass
                continue
            
            return {
                "target_id": target_id, "status": "failed", "error": err_msg,
                "download_time": time.time() - t_start, "preprocess_time": 0, "n_data": 0
            }

def main():
    parser = argparse.ArgumentParser(description="AstroTransit-GPU v1.3.0: High-performance Transit Search Platform")
    subparsers = parser.add_subparsers(dest="command")

    # 1. check
    subparsers.add_parser("check", help="Check GPU and CUDA availability")

    # 2. search
    parser_search = subparsers.add_parser("search", help="Search for transits in a single target")
    parser_search.add_argument("--target", type=str, required=True, help="Target TIC ID")
    parser_search.add_argument("--n-periods", type=int, default=5000)
    parser_search.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser_search.add_argument("--out", type=str, help="Save result to file")

    # 3. compare
    parser_compare = subparsers.add_parser("compare", help="Strict CPU vs GPU parity comparison")
    parser_compare.add_argument("--target", type=str, default="TIC 261136679")
    parser_compare.add_argument("--preset", choices=["standard", "large", "extreme"], default="standard")
    parser_compare.add_argument("--n-runs", type=int, default=5)
    parser_compare.add_argument("--out", type=str, default="comparison_report.md")

    # 4. inject
    parser_inject = subparsers.add_parser("inject", help="Run injection/recovery experiment")
    parser_inject.add_argument("--target", type=str, default="TIC 261136679")
    parser_inject.add_argument("--periods", type=str, default="2.0,5.0,10.0")
    parser_inject.add_argument("--depths", type=str, default="0.001,0.005,0.01")
    parser_inject.add_argument("--n-trials", type=int, default=5)
    parser_inject.add_argument("--out", type=str, default="injection_report.md")

    # 5. benchmark
    parser_bench = subparsers.add_parser("benchmark", help="Run reproducible benchmark from config")
    parser_bench.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser_bench.add_argument("--outdir", type=str, default="reports", help="Output directory")
    parser_bench.add_argument("--gpu-only", action="store_true", help="Skip CPU reference runs")

    # 6. batch
    parser_batch = subparsers.add_parser("batch", help="Batch processing of multiple targets")
    parser_batch.add_argument("--targets", type=str, required=True, help="CSV file with target IDs")
    parser_batch.add_argument("--out", type=str, default="batch_results.csv")
    parser_batch.add_argument("--workers", type=int, default=4, help="Number of download threads")
    parser_batch.add_argument("--resume", action="store_true", help="Skip targets already in output CSV")
    parser_batch.add_argument("--data-dir", type=str, default=None, help="Local directory for FITS files")
    parser_batch.add_argument("--n-periods", type=int, default=5000, help="Number of test periods")
    parser_batch.add_argument("--cpu", action="store_true", help="Use CPU (Astropy) instead of GPU")

    # build-cache command
    parser_cache = subparsers.add_parser("build-cache", help="Build consolidated binary cache for a sector")
    parser_cache.add_argument("--fits-dir", type=str, required=True, help="Directory containing FITS files")
    parser_cache.add_argument("--out-dir", type=str, required=True, help="Directory to save cache")
    parser_cache.add_argument("--workers", type=int, default=8, help="Number of parsing workers")

    # screen-sector command
    parser_screen = subparsers.add_parser("screen-sector", help="High-speed screening of a sector using cache")
    parser_screen.add_argument("--cache-dir", type=str, required=True, help="Directory containing built cache")
    parser_screen.add_argument("--out", type=str, default="screening_results.csv", help="Output results CSV")
    parser_screen.add_argument("--n-periods", type=int, default=5000, help="Number of test periods")
    parser_screen.add_argument("--precision", type=str, choices=["float32", "float64"], default="float32")
    parser_screen.add_argument("--blackwell", action="store_true", help="Force use of Blackwell-optimized V37 kernel")

    # 7. validate
    parser_validate = subparsers.add_parser("validate", help="Run official scientific validation pipeline")
    parser_validate.add_argument("--results", type=str, required=True, help="Path to screening results CSV")
    parser_validate.add_argument("--config", type=str, default="configs/validation_v39.yaml", help="Path to validation config")
    parser_validate.add_argument("--report", type=str, default="reports/OFFICIAL_VALIDATION_REPORT.md", help="Output report path")

    # 8. vet
    parser_vet = subparsers.add_parser("vet", help="Candidate Vetting Pipeline: Rank and evaluate candidates")
    parser_vet.add_argument("--results", type=str, required=True, help="Path to screening results CSV")
    parser_vet.add_argument("--cache-dir", type=str, help="Directory containing built sector cache (for plotting)")
    parser_vet.add_argument("--config", type=str, default="configs/vetting_v1.yaml", help="Path to vetting config")
    parser_vet.add_argument("--out", type=str, default="reports/vetting", help="Output directory for reports and plots")

    # 9. refine
    parser_refine = subparsers.add_parser("refine", help="Refine screening results: Detailed Top-K search on top targets")
    parser_refine.add_argument("--results", type=str, required=True, help="Path to initial screening results CSV")
    parser_refine.add_argument("--cache-dir", type=str, required=True, help="Directory containing built sector cache")
    parser_refine.add_argument("--config", type=str, default="configs/vetting_v1.yaml", help="Path to vetting config (for refinement rules)")
    parser_refine.add_argument("--out", type=str, default="screening_results_refined.csv", help="Output CSV path")
    parser_refine.add_argument("--top-k", type=int, default=5, help="Number of candidates per star")
    parser_refine.add_argument("--n-periods", type=int, default=10000, help="Number of periods for refined search")
    parser_refine.add_argument("--snr-threshold", type=float, help="Override SNR threshold from config")

    args = parser.parse_args()

    if args.command == "check":
        print(f"AstroTransit-GPU v1.3.0 Check")
        try:
            import cupy as cp
            cuda_available = cp.cuda.is_available()
            print(f"CUDA Available: {cuda_available}")
            if cuda_available:
                dev = cp.cuda.Device()
                print(f"Device: {dev.id} ({cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()})")
                print(f"Compute Capability: {dev.compute_capability}")
            else:
                print("Note: CUDA driver/GPU found, but CUDA is not available to CuPy.")
        except ImportError:
            print("CUDA Available: False (CuPy not installed)")
            print("Install GPU support with: pip install 'astrotransit-gpu[cuda12]'")

    elif args.command == "search":
        client = LightkurveClient()
        lc = client.download_lightcurve(args.target)
        lc_clean = clean_lightcurve(lc)
        t, y = to_arrays(lc_clean)
        dy = lc_clean.flux_err.value if hasattr(lc_clean, 'flux_err') else None
        
        periods = np.linspace(0.5, 20.0, args.n_periods)
        durations = np.linspace(0.01, 0.2, 5)
        
        print(f"Searching {args.target} ({len(t)} points, {args.n_periods} periods)...")
        model = BoxLeastSquaresGPU(t, y, dy=dy)
        dtype = np.float32 if args.precision == "float32" else np.float64
        res = model.power(periods, durations, dtype=dtype)
        
        print(f"\nBest Result:")
        print(f"  Period: {res.best_period:.6f} d")
        print(f"  T0: {res.best_t0:.4f}")
        print(f"  Depth: {res.best_depth:.4e}")
        print(f"  SNR/Power: {res.best_power:.2f}")

    elif args.command == "benchmark":
        import json
        from .validate.plot import plot_comparison, plot_folded_lc
        
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            
        target = cfg.get("target", "TIC 261136679")
        n_periods = cfg.get("n_periods", 5000)
        n_runs = cfg.get("timed_runs", 5)
        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)
        
        client = LightkurveClient()
        lc = client.download_lightcurve(target)
        lc_clean = clean_lightcurve(lc)
        t, y = to_arrays(lc_clean)
        dy = lc_clean.flux_err.value if hasattr(lc_clean, 'flux_err') else None
        
        periods = np.linspace(cfg.get("period_min", 0.5), cfg.get("period_max", 20.0), n_periods)
        durations = np.array(cfg.get("durations", [0.01, 0.0575, 0.105, 0.1525, 0.2]))
        
        print(f"Running Reproducible Benchmark: {cfg.get('benchmark_id', 'standard')}")
        
        # 1. CPU Runs
        cpu_times = []
        cpu_res = None
        if not args.gpu_only:
            print(f"Benchmarking CPU (Astropy)... This may take a while for large grids.")
            for i in range(n_runs):
                s = time.time()
                cpu_res = run_astropy_bls(t, y, dy=dy, periods=periods, durations=durations)
                cpu_times.append(time.time() - s)
        else:
            print(f"Skipping CPU benchmark (--gpu-only)")
            
        # 2. GPU Runs
        print(f"Benchmarking GPU (AstroTransit-GPU)...")
        model = BoxLeastSquaresGPU(t, y, dy=dy)
        gpu_times = []
        for i in range(n_runs):
            s = time.time()
            gpu_res = model.power(periods, durations, dtype=np.float32)
            gpu_times.append(time.time() - s)
            
        # 3. Analysis
        cpu_med = np.median(cpu_times) if cpu_times else 0.0
        gpu_med = np.median(gpu_times)
        cpu_p95 = np.percentile(cpu_times, 95) if cpu_times else 0.0
        gpu_p95 = np.percentile(gpu_times, 95)
        
        # Numerical Consistency Metrics (only if CPU ran)
        rmse = 0.0
        correlation = 0.0
        period_diff = 0.0
        if cpu_res is not None:
            rmse = np.sqrt(np.mean((cpu_res['power'] - gpu_res.power)**2))
            correlation = np.corrcoef(cpu_res['power'], gpu_res.power)[0, 1]
            period_diff = abs(cpu_res['period'] - gpu_res.best_period)
        
        # 4. JSON Output
        results_json = {
            "config": cfg,
            "statistics": {
                "cpu_median": cpu_med, "cpu_p95": cpu_p95,
                "gpu_median": gpu_med, "gpu_p95": gpu_p95,
                "speedup": cpu_med / gpu_med if cpu_med > 0 else 0.0,
                "rmse": rmse,
                "power_correlation": correlation,
                "best_period_diff": period_diff
            },
            "best_parameters": {
                "period_gpu": gpu_res.best_period,
                "period_cpu": cpu_res['period'] if cpu_res else None,
                "t0_gpu": gpu_res.best_t0,
                "depth_gpu": gpu_res.best_depth,
                "power_gpu": gpu_res.best_power
            }
        }
        with open(os.path.join(outdir, "benchmark.json"), "w") as f:
            json.dump(results_json, f, indent=4)
            
        # 5. Plots
        plot_comparison(periods, cpu_res['power'] if cpu_res else np.zeros_like(gpu_res.power), gpu_res.power, os.path.join(outdir, "periodogram_comparison.png"))
        plot_folded_lc(t, y, gpu_res.best_period, gpu_res.best_t0, os.path.join(outdir, "folded_lc.png"))
        
        # 6. Markdown Report
        cpu_info = f"| CPU (Astropy) | {cpu_med:.4f}s | {cpu_p95:.4f}s |" if not args.gpu_only else "| CPU (Astropy) | Skipped | Skipped |"
        speedup_info = f"| **Speedup** | **{cpu_med/gpu_med:.1f}x** | - |" if not args.gpu_only else "| **Speedup** | N/A | - |"
        
        consistency_info = ""
        if not args.gpu_only:
            consistency_info = f"""
- **Power Spectrum Correlation**: {correlation:.6f}
- **Power Spectrum RMSE**: {rmse:.6e}
- **Best Period Diff (CPU vs GPU)**: {period_diff:.6f} d
- """
            
        md = f"""# Benchmark Report: {cfg.get('benchmark_id', 'N/A')}
- **Target**: {target}
- **Data Points**: {len(t)}
- **Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Performance
| Backend | Median | P95 |
| :--- | :--- | :--- |
{cpu_info}
| GPU (Ours) | {gpu_med:.4f}s | {gpu_p95:.4f}s |
{speedup_info}

## Accuracy & Consistency
{consistency_info}
- **Best Period (GPU)**: {gpu_res.best_period:.6f} d

![Periodogram Comparison](./periodogram_comparison.png)
![Folded LC](./folded_lc.png)
"""
        with open(os.path.join(outdir, "report.md"), "w") as f:
            f.write(md)
            
        print(f"Benchmark completed. Results in {outdir}/")

    elif args.command == "compare":
        client = LightkurveClient()
        lc = client.download_lightcurve(args.target)
        lc_clean = clean_lightcurve(lc)
        t, y = to_arrays(lc_clean)
        dy = lc_clean.flux_err.value if hasattr(lc_clean, 'flux_err') else None
        
        n_periods = 5000
        if args.preset == "large": n_periods = 100000
        elif args.preset == "extreme": n_periods = 1000000
        
        periods = np.linspace(0.5, 20.0, n_periods)
        durations = np.linspace(0.01, 0.2, 5)
        
        print(f"Benchmarking CPU ({args.n_runs} runs)...")
        cpu_times = []
        for i in range(args.n_runs):
            s = time.time()
            cpu_res = run_astropy_bls(t, y, dy=dy, periods=periods, durations=durations)
            cpu_times.append(time.time() - s)
            
        print(f"Benchmarking GPU ({args.n_runs} runs)...")
        model = BoxLeastSquaresGPU(t, y, dy=dy)
        gpu_times = []
        for i in range(args.n_runs):
            s = time.time()
            gpu_res = model.power(periods, durations)
            gpu_times.append(time.time() - s)
            
        cpu_med, gpu_med = np.median(cpu_times), np.median(gpu_times)
        rmse = np.sqrt(np.mean((cpu_res['power'] - gpu_res.power)**2))
        
        report = f"# Parity Report: {args.target}\n"
        report += f"| Metric | CPU | GPU | Diff/Speedup |\n"
        report += f"| :--- | :--- | :--- | :--- |\n"
        report += f"| Runtime (Med) | {cpu_med:.4f}s | {gpu_med:.4f}s | **{cpu_med/gpu_med:.1f}x** |\n"
        report += f"| Best Period | {cpu_res['period']:.6f} | {gpu_res.best_period:.6f} | {abs(cpu_res['period']-gpu_res.best_period):.2e} |\n"
        report += f"| Spectrum RMSE | - | - | **{rmse:.4e}** |\n"
        
        with open(args.out, "w") as f:
            f.write(report)
        print(f"Comparison report written to {args.out}")

    elif args.command == "inject":
        client = LightkurveClient()
        lc = client.download_lightcurve(args.target)
        lc_clean = clean_lightcurve(lc)
        t, y = to_arrays(lc_clean)
        
        p_list = [float(x) for x in args.periods.split(",")]
        d_list = [float(x) for x in args.depths.split(",")]
        
        print(f"Running Injection Grid ({len(p_list)}x{len(d_list)} cells)...")
        results_df = run_injection_recovery_experiment(t, y, p_list, d_list, n_trials=args.n_trials)
        
        rec_map = calculate_recovery_map(results_df)
        print("\nRecovery Map:")
        print(rec_map)
        
        with open(args.out, "w") as f:
            f.write(f"# Injection Report\n\n{rec_map.to_markdown()}\n")
        print(f"Injection report written to {args.out}")

    elif args.command == "batch":
        from tqdm import tqdm
        
        if not os.path.exists(args.targets):
            print(f"Error: Target list {args.targets} not found.")
            return

        targets_df = pd.read_csv(args.targets)
        target_ids = targets_df['tic_id'].unique().tolist()
        
        # Resume logic
        processed_ids = set()
        file_exists = os.path.exists(args.out)
        if args.resume and file_exists:
            try:
                old_df = pd.read_csv(args.out)
                if 'tic_id' in old_df.columns and 'status' in old_df.columns:
                    # Only skip successful ones
                    processed_ids = set(old_df.loc[old_df['status'] == 'ok', 'tic_id'].unique().tolist())
                    print(f"Resuming: skipping {len(processed_ids)} successful targets.")
            except Exception:
                pass
        
        remaining_ids = [tid for tid in target_ids if tid not in processed_ids]
        
        if not remaining_ids:
            print("All targets already processed.")
            return

        periods = np.linspace(0.5, 20.0, args.n_periods)
        durations = np.linspace(0.01, 0.2, 5)
        
        backend_name = "CPU" if args.cpu else "GPU"
        print(f"Batch Analysis ({backend_name}): {len(remaining_ids)} remaining, {args.workers} threads, {args.n_periods} periods.")
        
        fieldnames = ["tic_id", "status", "period", "t0", "depth", "duration", "power", "n_data", 
                      "download_time", "preprocess_time", "gpu_time", "total_time", "error"]

        # Open file in append mode
        mode = 'a' if args.resume and file_exists else 'w'
        with open(args.out, mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if mode == 'w':
                writer.writeheader()
            
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_tid = {executor.submit(download_and_preprocess, tid, data_dir=args.data_dir): tid for tid in remaining_ids}
                
                with tqdm(total=len(remaining_ids), desc="Processing") as pbar:
                    for future in as_completed(future_to_tid):
                        item = future.result()
                        tid = item['target_id']
                        
                        res_row = {
                            "tic_id": tid, "status": item['status'],
                            "download_time": item['download_time'], "preprocess_time": item['preprocess_time'],
                            "n_data": item['n_data'], "error": item['error']
                        }
                        
                        if item['status'] == "ok":
                            try:
                                gpu_start = time.time()
                                if not args.cpu:
                                    model = BoxLeastSquaresGPU(item['time_array'], item['flux_array'], dy=item['dy_array'])
                                    res_obj = model.power(periods, durations)
                                    best_period = res_obj.best_period
                                    best_t0 = res_obj.best_t0
                                    best_depth = res_obj.best_depth
                                    best_power = res_obj.best_power
                                else:
                                    res_dict = run_astropy_bls(item['time_array'], item['flux_array'], dy=item['dy_array'], 
                                                               periods=periods, durations=durations)
                                    best_period = res_dict['period']
                                    best_t0 = res_dict['t0']
                                    best_depth = res_dict['depth']
                                    best_power = res_dict['power']
                                    
                                gpu_time = time.time() - gpu_start
                                
                                res_row.update({
                                    "period": best_period, "t0": best_t0,
                                    "depth": best_depth, "duration": float(res_obj.best_duration if not args.cpu else res_dict['duration']),
                                    "power": best_power,
                                    "gpu_time": gpu_time,
                                    "total_time": item['download_time'] + item['preprocess_time'] + gpu_time
                                })
                            except Exception as e:
                                res_row.update({"status": "failed", "error": f"Error: {str(e)}", "gpu_time": 0})
                        
                        writer.writerow(res_row)
                        csvfile.flush()
                        pbar.update(1)

        print(f"\nBatch analysis complete. Results saved to {args.out}")

    elif args.command == "build-cache":
        cache = SectorCache(args.out_dir)
        cache.build(args.fits_dir, workers=args.workers)

    elif args.command == "screen-sector":
        from .search.screener import GpuScreener
        import pandas as pd
        cache = SectorCache(args.cache_dir)
        data = cache.load()
        
        periods = np.linspace(0.5, 20.0, args.n_periods)
        durations = np.linspace(0.01, 0.2, 5)
        
        screener = GpuScreener(periods, durations, dtype=np.float32 if args.precision == "float32" else np.float64)
        results = screener.screen_sector(data, output_path=args.out, use_blackwell=args.blackwell)
        
        print(f"Screening complete. Final results saved to {args.out}")

    elif args.command == "validate":
        from .validate.official_validator import OfficialValidator
        validator = OfficialValidator(args.config)
        print(f"Running Official Validation: {args.results} ...")
        results_df = pd.read_csv(args.results)
        report_data = validator.run_validation(results_df)
        summary = report_data['summary']
        
        print("\n--- Validation Summary ---")
        print(f"  Total Targets: {summary['total_targets']:,}")
        print(f"  Completeness:  {summary['completeness']:.2%}")
        print(f"  New Candidates: {summary['new_candidates']:,}")
        print(f"  FPR:           {summary['fpr']:.2%}")
        
        # Save Markdown Report
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(f"# Official Validation Report\n\nGenerated: {pd.Timestamp.now()}\n\n")
            f.write(pd.Series(summary).to_markdown())
            f.write("\n")
        print(f"\nReport saved to: {args.report}")

    elif args.command == "vet":
        from .vet.pipeline import run_vetting_pipeline
        run_vetting_pipeline(
            results_csv=args.results,
            cache_dir=args.cache_dir,
            config_path=args.config,
            out_dir=args.out
        )

    elif args.command == "refine":
        from .search.refiner import CandidateRefiner
        import yaml
        
        # 1. Load full config safely
        full_cfg = {}
        if args.config and os.path.exists(args.config):
            try:
                with open(args.config, 'r') as f:
                    full_cfg = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load config {args.config}: {e}")
        
        # 2. Extract specific sections
        refine_cfg = full_cfg.get('refinement', {})
        catalog_cfg = full_cfg.get('catalogs', {})
        
        # 3. CLI override
        if args.snr_threshold is not None:
            refine_cfg['snr_threshold'] = args.snr_threshold

        # 4. Execute refinement
        refiner = CandidateRefiner(cache_dir=args.cache_dir, n_periods=args.n_periods)
        refiner.refine_from_csv(
            input_csv=args.results,
            output_csv=args.out,
            config=refine_cfg,
            toi_path=catalog_cfg.get('toi_catalog'),
            eb_path=catalog_cfg.get('eb_catalog'),
            top_k=args.top_k
        )

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
