import argparse
import sys
import numpy as np
import time
import yaml
import pandas as pd
from .data.lightkurve_client import LightkurveClient
from .preprocess.clean import clean_lightcurve, to_arrays
from .search.api import BoxLeastSquaresGPU
from .search.cpu_reference_bls import run_astropy_bls
from .inject.grid import run_injection_recovery_experiment, calculate_recovery_map

def main():
    parser = argparse.ArgumentParser(description="AstroTransit-GPU v1.0: High-performance Transit Search Platform")
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

    # 6. batch
    parser_batch = subparsers.add_parser("batch", help="Batch processing of multiple targets")
    parser_batch.add_argument("--targets", type=str, required=True, help="CSV file with target IDs")

    args = parser.parse_args()

    if args.command == "check":
        print(f"AstroTransit-GPU v1.0 Check")
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
        import os
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
        for i in range(n_runs):
            s = time.time()
            cpu_res = run_astropy_bls(t, y, dy=dy, periods=periods, durations=durations)
            cpu_times.append(time.time() - s)
            
        # 2. GPU Runs
        model = BoxLeastSquaresGPU(t, y, dy=dy)
        gpu_times = []
        for i in range(n_runs):
            s = time.time()
            gpu_res = model.power(periods, durations, dtype=np.float32)
            gpu_times.append(time.time() - s)
            
        # 3. Analysis
        cpu_med, gpu_med = np.median(cpu_times), np.median(gpu_times)
        cpu_p95, gpu_p95 = np.percentile(cpu_times, 95), np.percentile(gpu_times, 95)
        
        # Numerical Consistency Metrics
        rmse = np.sqrt(np.mean((cpu_res['power'] - gpu_res.power)**2))
        correlation = np.corrcoef(cpu_res['power'], gpu_res.power)[0, 1]
        period_diff = abs(cpu_res['period'] - gpu_res.best_period)
        
        # 4. JSON Output
        results_json = {
            "config": cfg,
            "statistics": {
                "cpu_median": cpu_med, "cpu_p95": cpu_p95,
                "gpu_median": gpu_med, "gpu_p95": gpu_p95,
                "speedup": cpu_med / gpu_med,
                "rmse": rmse,
                "power_correlation": correlation,
                "best_period_diff": period_diff
            },
            "best_parameters": {
                "period_gpu": gpu_res.best_period,
                "period_cpu": cpu_res['period'],
                "t0_gpu": gpu_res.best_t0,
                "depth_gpu": gpu_res.best_depth,
                "power_gpu": gpu_res.best_power
            }
        }
        with open(os.path.join(outdir, "benchmark.json"), "w") as f:
            json.dump(results_json, f, indent=4)
            
        # 5. Plots
        plot_comparison(periods, cpu_res['power'], gpu_res.power, os.path.join(outdir, "periodogram_comparison.png"))
        plot_folded_lc(t, y, gpu_res.best_period, gpu_res.best_t0, os.path.join(outdir, "folded_lc.png"))
        
        # 6. Markdown Report
        md = f"""# Benchmark Report: {cfg.get('benchmark_id', 'N/A')}
- **Target**: {target}
- **Data Points**: {len(t)}
- **Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Performance
| Backend | Median | P95 |
| :--- | :--- | :--- |
| CPU (Astropy) | {cpu_med:.4f}s | {cpu_p95:.4f}s |
| GPU (Ours) | {gpu_med:.4f}s | {gpu_p95:.4f}s |
| **Speedup** | **{cpu_med/gpu_med:.1f}x** | - |

## Accuracy & Consistency
- **Power Spectrum Correlation**: {correlation:.6f}
- **Power Spectrum RMSE**: {rmse:.6e}
- **Best Period Diff (CPU vs GPU)**: {period_diff:.6f} d
- **Best Period (GPU)**: {gpu_res.best_period:.6f} d

![Periodogram Comparison](./periodogram_comparison.png)
![Folded LC](./folded_lc.png)
"""
        with open(os.path.join(outdir, "report.md"), "w") as f:
            f.write(md)
            
        print(f"Benchmark completed. Results in {outdir}/")

    elif args.command == "compare":
        # (Internal logic similar to previous but using BoxLeastSquaresGPU)
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
        
        # Simple text heatmap in stdout
        rec_map = calculate_recovery_map(results_df)
        print("\nRecovery Map:")
        print(rec_map)
        
        with open(args.out, "w") as f:
            f.write(f"# Injection Report\n\n{rec_map.to_markdown()}\n")
        print(f"Injection report written to {args.out}")

    elif args.command == "batch":
        print(f"Batch processing not yet implemented in v1.0 refactor.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
