# AstroTransit-GPU (v1.4.0)

[![CI](https://github.com/ubunbun18/Astro_Transit_GPU/actions/workflows/ci.yml/badge.svg)](https://github.com/ubunbun18/Astro_Transit_GPU/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-1.4.0-blue.svg)](./CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**AstroTransit-GPU** is a high-performance, researcher-grade platform for exoplanet transit discovery. It bridges the gap between massive-scale screening and rigorous scientific validation by providing two specialized CUDA engines: the high-throughput **Fast Engine (V41)** and the high-precision **Parity Engine (V42)**.

---

## 🌟 Key Features

| Feature | Description |
| :--- | :--- |
| **🚀 High-Speed Screening** | Screen a full TESS sector (15,881 targets) in **~73 seconds** (V41, 5,000 periods, measured). |
| **🎯 Numerical Fidelity** | Proven correlation of **0.999997** and ΔT0 of **1.43e-08 d** vs Astropy (V42). |
| **🛠️ Dual Engine** | Choose between **High-Throughput Screening** and **Scientific Validation**. |
| **📦 Survey Pipeline** | Consolidated Binary Cache system to eliminate disk I/O bottlenecks. |
| **🧪 Validated** | Verified against NASA TOI catalog with **38.75% recovery rate** (Sector 1 FFI). |
| **💻 Modern API** | Astropy-compatible `BoxLeastSquaresGPU` class for seamless Python integration. |

---

## 🏎️ Dual Engine Architecture

AstroTransit-GPU v1.4.0 introduces a dual-engine strategy to cover the entire transit search workflow.

### 1. Fast Engine (V41) — *For Screening*
Designed for massive-scale surveys. Tuned for NVIDIA Blackwell (and RTX 40-series) architectures.
- **Throughput**: **234 LC/s** (5,000 periods, 15,000 data points).
- **Optimization**: Branchless boundary processing, algebraic FDIV reduction, 100% Warp Occupancy.
- **Caveat**: Uses parallel atomic operations; non-deterministic with tiny numerical drift (5.72e-06).

### 2. Parity Engine (V42) — *For Validation*
Designed for scientific reproducibility. Bit-level identical operation order to CPU reference.
- **Accuracy**: **Correlation = 0.999997** vs Astropy; Numerical Drift = **0.000000** (Perfect Determinism).
- **Throughput**: **48 LC/s** (5,000 periods, 15,000 data points).
- **Best for**: Final candidate verification and generating paper-quality results.

---

## 🚀 Quick Start

### Python API Integration
```python
from astrotransit_gpu import BoxLeastSquaresGPU

# Initialize with time, flux, and optional error
model = BoxLeastSquaresGPU(t, y, dy=dy)

# High-Throughput Mode (Default)
res_fast = model.power(periods, durations, method="fast")

# Scientific Parity Mode (Astropy-Compatible)
res_exact = model.power(periods, durations, method="parity")
```

### Command Line Interface
```bash
# Diagnostic check
astrotransit-gpu check

# Search a target using the Parity mode
astrotransit-gpu search --target "TIC 261136679" --method parity

# Run a sector-wide screening using the Fast engine
astrotransit-gpu screen-sector --cache-dir ./data_cache --n-periods 5000
```

---

## 📊 Measured Benchmarks (v1.4.0)

**Hardware**: AMD Ryzen 7 9700X + NVIDIA GeForce RTX 5060 Ti (Blackwell)
**Conditions**: 5,000 periods, 15,000 data points per target

| Kernel | Throughput (LC/s) | Speedup (vs CPU) | Correlation (vs Astropy) |
| :--- | :--- | :--- | :--- |
| CPU (Astropy) | 3.97 | 1.0x | 1.000000 |
| **V41 (Fast)** | **234.07** | **58.9x** | 0.967864 |
| **V42 (Parity)** | **48.05** | **12.1x** | **0.999997** |

**Full Sector (15,881 targets) Marathon Results**:
- V41 (Fast): **72.78 sec (1.21 min)**
- V42 (Parity): **362.95 sec (6.05 min)**
- CPU (Astropy): **~4,000 sec (66.67 min)** (Extrapolated from 3.97 LC/s)

---

## 📄 Documentation
- [CLI Reference](./docs/CLI_REFERENCE.md)
- [Kernel Selection Guide](./docs/KERNEL_GUIDE_V39_V42_JP.md)
- [Benchmark Report V3](./BENCHMARK_REPORT_V3.md)
- [Sector Cache Design](./docs/SECTOR_CACHE_V2.md)

---
© 2026 AstroTransit-GPU Team. Licensed under MIT.