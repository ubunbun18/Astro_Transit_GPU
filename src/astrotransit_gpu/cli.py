import argparse
import sys
from . import __version__

from .data.lightkurve_client import LightkurveClient
from .preprocess.clean import clean_lightcurve, to_arrays
from .report.plots import plot_periodogram, plot_folded_lightcurve
from .report.json_report import save_json_report
from .search.gpu_bls import run_gpu_bls, get_top_k_candidates
import yaml
import numpy as np
from .inject.grid import run_injection_recovery_experiment, calculate_recovery_map
from .data.exoplanet_archive import ExoplanetArchiveClient
from .validate.known_planets import run_batch_known_search, generate_batch_report_md
from .validate.match import match_candidate
from .report.markdown_report import generate_markdown_report
import os

def main():
    parser = argparse.ArgumentParser(description="AstroTransit-GPU: GPU-accelerated transit search platform")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # 'known' command
    known_parser = subparsers.add_parser("known", help="Redetect known planets")
    known_parser.add_argument("--target", type=str, required=True, help="Target ID (e.g. 'TIC 261136679')")
    known_parser.add_argument("--mission", choices=["tess", "kepler"], default="tess")
    known_parser.add_argument("--true-p", type=float, help="True period if known")
    known_parser.add_argument("--true-t0", type=float, help="True t0 if known")
    known_parser.add_argument("--out", type=str, default="report.md", help="Output path")
    
    # 'batch' command
    batch_parser = subparsers.add_parser("batch", help="Batch process known targets")
    batch_parser.add_argument("--n-targets", type=int, default=10, help="Number of targets to fetch")
    batch_parser.add_argument("--min-depth", type=float, default=500.0, help="Minimum catalog depth in ppm")
    batch_parser.add_argument("--out", type=str, default="batch_report.md", help="Output path")
    
    # 'inject-run' command
    inject_parser = subparsers.add_parser("inject-run", help="Run injection/recovery experiment")
    inject_parser.add_argument("--target", type=str, required=True, help="Base target ID")
    inject_parser.add_argument("--periods", type=str, default="2.0,5.0,10.0", help="Comma-separated periods to inject")
    inject_parser.add_argument("--depths", type=str, default="0.001,0.003,0.01", help="Comma-separated depths to inject")
    inject_parser.add_argument("--n-trials", type=int, default=3, help="Trials per grid cell")
    inject_parser.add_argument("--out", type=str, default="injection_recovery_report.md", help="Output Markdown report path")

    # 'run-config' command
    run_cfg_parser = subparsers.add_parser("run-config", help="Run search using a YAML config file")
    run_cfg_parser.add_argument("config", type=str, help="Path to YAML config file")
    
    # 0. Check command
    parser_check = subparsers.add_parser("check", help="Check environment and GPU availability")
    
    # 1. Compare command
    parser_compare = subparsers.add_parser("compare", help="Compare CPU (Astropy) vs GPU (CUDA) performance and accuracy")
    parser_compare.add_argument("--target", type=str, default="TIC 261136679", help="Target TIC ID")
    parser_compare.add_argument("--out", type=str, default="comparison_report.md", help="Output Markdown report path")
    parser_compare.add_argument("--n-periods", type=int, default=5000, help="Number of periods in grid")
    
    args = parser.parse_args()
    
    if args.command == "check":
        print("--- AstroTransit-GPU Environment Check ---")
        import cupy as cp
        print(f"CuPy version: {cp.__version__}")
        try:
            device = cp.cuda.Device(0)
            print(f"GPU Device 0: {device.attributes['Name'] if 'Name' in device.attributes else 'Available'}")
            print(f"Compute Capability: {device.compute_capability}")
            print(f"Total Memory: {device.mem_info[1] / 1024**3:.2f} GB")
            print("Status: SUCCESS - GPU is ready.")
        except Exception as e:
            print(f"Status: FAILED - {e}")
        sys.exit(0)

    if args.command == "known":
        client = LightkurveClient()
        lc = client.download_lightcurve(args.target, mission=args.mission)
        lc_clean = clean_lightcurve(lc)
        time_arr, flux_arr = to_arrays(lc_clean)
        
        periods = np.linspace(0.5, 20.0, 1000)
        durations = np.linspace(0.01, 0.2, 5)
        
        print(f"Searching for transits in {args.target}...")
        results = run_gpu_bls(time_arr, flux_arr, periods, durations)
        
        if args.true_p:
            m = match_candidate(
                results['best_period'], 
                results['best_t0'], 
                args.true_p, 
                args.true_t0,
                require_t0=args.true_t0 is not None
            )
            results['match'] = m
            
        # Extract Top-K
        results['top_candidates'] = get_top_k_candidates(results, k=5)
            
        generate_markdown_report(results, args.out)
        
        # New features
        save_json_report(results, args.out.replace(".md", ".json"))
        plot_periodogram(periods, results['power'].get(), results['best_period'], "periodogram.png")
        plot_folded_lightcurve(time_arr, flux_arr, results['best_period'], results['best_t0'], results['best_duration'], "folded_lc.png")
        
        print(f"Extended results generated: {args.out}, JSON, and plots.")
    elif args.command == "batch":
        from .search.parallel_search import run_hyper_batch_search
        print(f"Fetching {args.n_targets} targets from NASA Exoplanet Archive...")
        targets = ExoplanetArchiveClient.get_toi_targets(n_targets=args.n_targets, min_depth_ppm=args.min_depth)
        
        target_ids = [t['target_id'] for t in targets]
        print(f"Starting Hyper-Speed Async Batch Search for {len(target_ids)} stars...")
        
        results = run_hyper_batch_search(target_ids)
        
        # Post-process results into a DataFrame for report
        # Match results with targets by target_id since they are asynchronous
        target_map = {t['target_id']: t for t in targets}
        summary_results = []
        for r in results:
            tid = r['target_id']
            t = target_map.get(tid)
            if 'error' in r:
                summary_results.append({'target_id': tid, 'is_recovered': False, 'error': r['error']})
            elif t:
                m = match_candidate(r['best_period'], r['best_t0'], t['period'], t['t0'])
                summary_results.append({
                    'target_id': tid,
                    'true_period': t['period'],
                    'detected_period': r['best_period'],
                    'is_recovered': m['is_match'],
                    'match_type': m['match_type'],
                    'snr': r['snr']
                })
        
        import pandas as pd
        df_results = pd.DataFrame(summary_results)
        generate_batch_report_md(df_results, args.out)
        print(f"Hyper-Batch report generated: {args.out}")
    elif args.command == "inject-run":
        client = LightkurveClient()
        print(f"Downloading base light curve: {args.target}")
        lc = client.download_lightcurve(args.target)
        lc_clean = clean_lightcurve(lc)
        time_arr, flux_arr = to_arrays(lc_clean)
        
        p_list = [float(p) for p in args.periods.split(",")]
        d_list = [float(d) for d in args.depths.split(",")]
        
        print(f"Starting Injection/Recovery Grid ({len(p_list)}x{len(d_list)} cells, {args.n_trials} trials each)...")
        df_results = run_injection_recovery_experiment(time_arr, flux_arr, p_list, d_list, n_trials=args.n_trials)
        
        recovery_map = calculate_recovery_map(df_results)
        
        # Save results
        md = "# Injection/Recovery Recovery Map\n\n"
        md += "Values represent recovery probability (0.0 to 1.0).\n\n"
        md += recovery_map.to_markdown()
        md += "\n\n## Detailed Trials\n\n"
        md += df_results.head(20).to_markdown(index=False)
        
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Injection/Recovery report generated: {args.out}")
    elif args.command == "run-config":
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        
        target = cfg.get('target')
        mission = cfg.get('mission', 'tess')
        client = LightkurveClient()
        lc = client.download_lightcurve(target, mission=mission)
        lc_clean = clean_lightcurve(lc)
        time_arr, flux_arr = to_arrays(lc_clean)
        
        search_cfg = cfg.get('search', {})
        periods = np.linspace(
            search_cfg.get('period_min', 0.5),
            search_cfg.get('period_max', 20.0),
            search_cfg.get('n_periods', 5000)
        )
        durations = np.linspace(0.01, 0.2, 5)
        
        results = run_gpu_bls(time_arr, flux_arr, periods, durations)
        
        # Generate plots
        plot_periodogram(periods, results['power'].get(), results['best_period'], "periodogram.png")
        plot_folded_lightcurve(time_arr, flux_arr, results['best_period'], results['best_t0'], results['best_duration'], "folded_lc.png")
        
        # Save JSON
        save_json_report(results, "results.json")
        print("Run completed from config. Plots and JSON generated.")
    elif args.command == "compare":
        from .search.cpu_reference_bls import run_astropy_bls
        import time
        client = LightkurveClient()
        lc = client.download_lightcurve(args.target)
        lc_clean = clean_lightcurve(lc)
        time_arr, flux_arr = to_arrays(lc_clean)
        
        periods = np.linspace(0.5, 20.0, args.n_periods)
        durations = np.linspace(0.01, 0.2, 5)
        
        print(f"Running CPU BLS with {len(periods)} periods...")
        start = time.time()
        cpu_res = run_astropy_bls(time_arr, flux_arr, periods=periods, durations=durations)
        cpu_time = time.time() - start
        
        print(f"Running GPU BLS with {len(periods)} periods...")
        start = time.time()
        gpu_res = run_gpu_bls(time_arr, flux_arr, periods, durations)
        gpu_time = time.time() - start
        
        # Report
        md = "# CPU vs GPU Numerical Comparison\n\n"
        md += f"| Metric | CPU (Astropy) | GPU (Custom CUDA) | Difference |\n"
        md += f"| --- | --- | --- | --- |\n"
        md += f"| Best Period | {cpu_res['period']:.6f} | {gpu_res['best_period']:.6f} | {abs(cpu_res['period']-gpu_res['best_period']):.6e} |\n"
        md += f"| Best T0 | {cpu_res['t0']:.6f} | {gpu_res['best_t0']:.6f} | {abs(cpu_res['t0']-gpu_res['best_t0']):.6e} |\n"
        md += f"| Runtime | {cpu_time:.4f}s | {gpu_time:.4f}s | x{cpu_time/gpu_time:.1f} faster |\n\n"
        
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Comparison report generated: {args.out}")
    else:
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
