# AstroTransit-GPU 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**AstroTransit-GPU** is a CUDA-accelerated platform for planet transit discovery from time-series photometry (e.g., TESS, Kepler). By combining custom CUDA kernels with an asynchronous processing pipeline, it maximizes discovery throughput for large-scale exoplanet surveys.

## 🔬 Technical Features

-   **Parallel BLS Kernel**: 
    - **Period Tiling**: Processes multiple trial periods in parallel during a single memory pass to improve bandwidth utilization.
    - **Parallel Scan & Search**: Utilizes shared-memory prefix sums for high-speed sliding window search across GPU threads.
-   **Asynchronous Pipeline**:
    - Overlaps CPU preprocessing (download, cleaning) and GPU computations using `ProcessPoolExecutor`.
-   **Validated Numerical Parity**:
    - Rigorously compared against `astropy.timeseries.BoxLeastSquares` on identical search grids to ensure scientific integrity.

## 📊 Performance & Reliability

AstroTransit-GPU demonstrates significant throughput gains over CPU-based implementations (Astropy), especially as the period search density increases.

| Search Scale | Period Grid Size | Speedup Factor (vs. CPU) |
| :--- | :--- | :--- |
| **Standard** | 5,000 | ~7.5x |
| **Large** | 100,000 | ~79x |
| **Extreme** | 1,000,000 | **>130x** |

> [!NOTE]
> For detailed measurement conditions, numerical consistency reports, and reproduction commands, please refer to [BENCHMARK_REPORT.md](./BENCHMARK_REPORT.md).

## 🚀 Installation

```bash
git clone https://github.com/ubunbun18/Astro_Transit_GPU.git
cd Astro_Transit_GPU
pip install .
```

## 🛠️ CLI Commands

| Command | Description |
| :--- | :--- |
| `check` | Diagnose GPU availability and CUDA environment. |
| `compare` | Direct performance and accuracy comparison between CPU and GPU. |
| `known` | Search and detailed reporting on specific known targets. |
| `batch` | Mass-download and analyze targets from the NASA archives. |
| `inject-run` | Perform injection/recovery experiments to evaluate detection limits. |
| `run-config` | Execute reproducible experiments using YAML configuration files. |

Detailed options and usage examples are available in the [CLI Reference](./BENCHMARK_REPORT.md).

## 📋 Reproducing Benchmarks

To reproduce performance results on your hardware, use the following command:

```bash
# Compare using Standard resolution
astrotransit-gpu compare --preset standard
```

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Scaling Exoplanet Discovery with Reliability.*