# AstroTransit-GPU CLI Reference (v1.0.0)

AstroTransit-GPU provides a powerful command-line interface designed for high-throughput exoplanet transit search and reproducible scientific validation.

---

## 🚀 General Specifications
- **Units**: Periods are in **days**, and transit depths are **relative** to the stellar flux.
- **Target Identifiers**: Supports TIC IDs (e.g., `TIC 123456`) and KIC IDs.

---

## 🛠️ Command Details

### 1. `check` — Environment Diagnostics
Checks for CUDA availability and provides GPU hardware specifications.

- **Usage**:
  ```bash
  astrotransit-gpu check
  ```
- **Output**:
  - CUDA availability status.
  - Device name and Compute Capability.

---

### 2. `search` — Single Target Search
Perform a fast BLS search directly on a specific target.

- **Arguments**:
  - `--target` (Required): Target identifier (e.g., "TIC 261136679").
  - `--n-periods` (Default: 5000): Number of periods to search in the grid.
  - `--precision` (Default: `float32`): Computation precision (`float32` or `float64`).
  - `--out`: Path to save results in JSON format.
- **Usage**:
  ```bash
  astrotransit-gpu search --target "TIC 261136679" --n-periods 10000
  ```

---

### 3. `compare` — Performance & Parity Validation
Directly compares CPU (Astropy) and GPU results to verify numerical accuracy and speedup factors.

- **Arguments**:
  - `--target` (Default: "TIC 261136679"): Target used for validation.
  - `--preset`: Search scale preset.
    - `standard`: 5,000 periods (Default)
    - `large`: 100,000 periods
    - `extreme`: 1,000,000 periods
  - `--n-runs` (Default: 5): Number of trials to calculate median runtime.
  - `--out` (Default: `comparison_report.md`): Path to the parity report.
- **Usage**:
  ```bash
  astrotransit-gpu compare --preset large --n-runs 3
  ```

---

### 4. `inject` — Injection/Recovery Experiment
Evaluate detection limits by injecting synthetic transit signals into real light curve data.

- **Arguments**:
  - `--target`: Base light curve data.
  - `--periods` (Default: "2.0,5.0,10.0"): Comma-separated list of periods to inject.
  - `--depths` (Default: "0.001,0.005,0.01"): Comma-separated list of transit depths.
  - `--n-trials` (Default: 5): Number of trials per grid cell.
  - `--out` (Default: `injection_report.md`): Path to the recovery map report.
- **Usage**:
  ```bash
  astrotransit-gpu inject --periods "2.5,5.0,7.5" --depths "0.001,0.01" --n-trials 10
  ```

---

### 5. `benchmark` — Reproducible Benchmark Execution
Automates scientific report generation based on YAML configuration files.

- **Arguments**:
  - `--config` (Required): Path to the YAML configuration file.
  - `--outdir` (Default: `reports`): Directory to save reports, plots, and JSON data.
- **YAML Config Example**:
  ```yaml
  benchmark_id: "Survey_Beta_Run"
  target: "TIC 261136679"
  period_min: 0.5
  period_max: 20.0
  n_periods: 5000
  timed_runs: 5
  durations: [0.01, 0.05, 0.1]
  ```
### `benchmark`
Generate a reproducible performance report.

```bash
astrotransit-gpu benchmark --config config.yaml [--outdir reports] [--gpu-only]
```

- `--config`: Path to the YAML configuration file.
- `--outdir`: Directory to save the report (default: `reports`).
- `--gpu-only`: Skip CPU (Astropy) reference runs. Recommended for massive grids.

---

### 6. `batch` — Batch Analysis (Beta)
### `batch`
Batch process multiple targets using asynchronous I/O to minimize network bottlenecks.

```bash
astrotransit-gpu batch --targets targets.csv [--out results.csv] [--workers 4] [--resume]
```

- `--targets`: CSV file containing a `tic_id` column.
- `--out`: Output CSV filename (default: `batch_results.csv`).
- `--workers`: Number of parallel threads for download and preprocessing (default: 4).
- `--resume`: Skip targets already marked as `ok` in the output CSV.
- **Robustness**: Includes automated detection and cleanup of corrupted FITS cache files with automatic retry.
- **Usage**:
  ```bash
  astrotransit-gpu batch --targets candidate_list.csv
  ```

### 7. `build-cache` — Sector Cache Construction
Parses thousands of FITS files, performs preprocessing (NaN removal, normalization), and consolidates them into a single, high-speed flat binary NPZ file that the GPU can ingest efficiently.

- **Arguments**:
  - `--fits-dir` (Required): Directory containing raw FITS files downloaded from MAST.
  - `--out-dir` (Required): Directory to save the cache files.
  - `--workers` (Default: 8): Number of CPU workers for FITS parsing.
- **Usage**:
  ```bash
  astrotransit-gpu build-cache --fits-dir data/tess_sector1 --out-dir data/sector1_cache
  ```

---

### 8. `screen-sector` — Ultra-Fast Sector Screening
Utilizes the consolidated cache to analyze all targets in a sector at maximum GPU throughput. By eliminating disk I/O bottlenecks, it achieves massive screening speeds.

- **Arguments**:
  - `--cache-dir` (Required): Directory containing the built cache.
  - `--n-periods` (Default: 5000): Density of the period grid.
  - `--precision` (Default: `float32`): Computation precision.
  - `--out` (Default: `screening_results.csv`): Path to save results.
- **Usage**:
  ```bash
  # Screens 16,000 targets with 100k periods in ~30 minutes
  astrotransit-gpu screen-sector --cache-dir data/sector1_cache --n-periods 100000
  ```

---

### 💡 Tips: Bulk Downloads
For large datasets, we provide a utility script to parse official MAST curl scripts and download data with high concurrency.

```bash
python scripts/bulk_download_sector.py --script path/to/mast_curl_script.sh --outdir data/tess_sector1 --threads 50
```
This is significantly faster than standard `batch` downloads for full-sector datasets.

### 1. Memory Management
Large `n_bins` or `N_TILE` can exceed GPU shared memory limits (typically 48KB).
- **Tip**: Keep `n_bins` under 500 or use `float32` to minimize memory footprint.

### 2. High-Precision Mode
Use `--precision float64` for datasets with baselines > 100 days where phase precision is vital for narrow transits.

### 3. Pipeline Integration
The `benchmark` command generates a `benchmark.json` file, which is designed for easy parsing by external data analysis tools.
