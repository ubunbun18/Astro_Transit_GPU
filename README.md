# AstroTransit-GPU (v1.0.0)

[![CI](https://github.com/ubunbun18/Astro_Transit_GPU/actions/workflows/ci.yml/badge.svg)](https://github.com/ubunbun18/Astro_Transit_GPU/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](./CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**AstroTransit-GPU** is a high-performance, researcher-grade platform for exoplanet transit discovery. It accelerates the Box Least Squares (BLS) algorithm using custom CUDA kernels while ensuring rigorous numerical parity with Astropy. It is designed for reproducible scientific analysis, featuring automated benchmarking and injection/recovery suites.

## 🌟 Key Features

- **Blazing Fast & Accurate**: Achieves >100x throughput vs. Astropy while maintaining high [correlation (>0.95)](./BENCHMARK_REPORT.md#numerical-parity).
- **Survey-Scale Pipeline**: Uses a **Consolidated Sector Cache** system to eliminate disk I/O bottlenecks.
- **Ultra-High Throughput**: Capable of screening an entire TESS sector (16k targets) with **100k periods in ~30 minutes** (~9 targets/sec).
- **Astropy-Compatible API**: Features the `BoxLeastSquaresGPU` class for easy integration.
- **Production-Ready & Robust**: Automated detection and cleanup of corrupt FITS cache with retry logic.
- **Full Precision Support**: Supports weighted observations (`flux_err`) and `float64` precision.
- **Extreme Reproducibility**: Uses YAML-based configurations, fixed seeds, and robust statistical metrics (Median/P95).

## 🚀 Installation

While the package can be installed on CPU-only environments, a CUDA-enabled GPU and CuPy are required for acceleration.

```bash
# Clone and install in editable mode (recommended)
git clone https://github.com/ubunbun18/Astro_Transit_GPU.git
cd Astro_Transit_GPU
pip install -e ".[cuda12,benchmark]"
```

## 🛠️ Quick Start (Python API)

```python
from astrotransit_gpu import BoxLeastSquaresGPU
import numpy as np

# Prepare your light curve data
t = np.linspace(0, 10, 5000)
y = np.ones_like(t)  # flux
dy = np.ones_like(t) * 0.001 # error (optional)

# Initialize the model (Astropy-compatible)
model = BoxLeastSquaresGPU(t, y, dy=dy)

# Run the search
periods = np.linspace(0.5, 20.0, 10000)
durations = [0.05, 0.1, 0.15]
results = model.power(periods, durations, n_bins=500)

print(f"Best Period: {results.best_period:.4f} days")
print(f"Best Power (SNR): {results.best_power:.2f}")
```

## 💻 CLI Commands

| Command | Description |
| :--- | :--- |
| `check` | Diagnose GPU availability and CUDA environment. |
| `compare` | Direct performance and accuracy comparison with CPU. Supports `--preset`. |
| `inject` | Run injection/recovery experiments and generate Recovery Heatmaps. |
| `benchmark` | Automated report generation from YAML configuration files. |
| `search` | Quick search and result visualization for a single target. |
| `batch` | Mass analysis for target lists. |
| `build-cache` | Consolidate thousands of FITS files into a high-speed flat binary. |
| `screen-sector` | High-speed sector-wide screening using consolidated cache. |

For detailed arguments and usage examples, please refer to the [CLI Reference (docs/CLI_REFERENCE.md)](./docs/CLI_REFERENCE.md).

## 📊 Benchmarks

For detailed performance metrics, hardware configurations, and reproduction steps, please refer to [BENCHMARK_REPORT.md](./BENCHMARK_REPORT.md).

## 📖 Limitations & Notes

- The GPU backend currently utilizes a Phase Binning algorithm for extreme throughput.
- Default precision is `float32`. Use `float64` for ultra-long baseline data where phase precision is critical.
- This package is intended for rapid candidate screening and does not include full MCMC fitting.

## 📄 Citation

If you use this software in your research, please cite it using the metadata provided in [CITATION.cff](./CITATION.cff).

## ⚖️ License

Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.