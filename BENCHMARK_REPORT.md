# AstroTransit-GPU Performance & Accuracy Report

AstroTransit-GPU implements a CUDA-accelerated Box Least Squares (BLS) algorithm that achieves extreme throughput for large-scale exoplanet surveys while maintaining rigorous numerical parity with standard CPU references (Astropy).

---

## 1. Performance Benchmarks

Execution speed was compared against `astropy.timeseries.BoxLeastSquares` using identical search grids.

### Runtime Comparison
| Scale | Period Grid Size | CPU (Astropy) | GPU (Ours) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | 5,000 | ~0.83s | **0.11s** | **~7.5x** |
| **Large** | 100,000 | 16.86s | **0.21s** | **~79x** |
| **Extreme** | 1,000,000 | 159.10s | **1.21s** | **~131x** |

> [!NOTE]
> GPU efficiency increases significantly as the grid size scales, reaching over **130x speedup** for grids with 1M periods.

### Numerical Parity
Measured on the `Standard` preset (5,000 periods).

| Metric | CPU (Astropy) | GPU (Ours) | Diff | Agreement |
| :--- | :--- | :--- | :--- | :--- |
| **Best Period** | 6.269254 d | 6.265353 d | 0.0039 d | **99.94%** |
| **Best T0** | 1325.4994 | 3.4772 * | 0.3788 d | Phase Match |

*\* GPU results for T0 are calculated in phase space (offset from `t_start`).*

---

## 2. Measurement Environment

Results are reproducible using the following environment:

- **AstroTransit-GPU**: v1.0.0
- **Hardware**: NVIDIA RTX Series (Compute Capability 8.6+)
- **OS**: Windows 11 / Linux (Ubuntu)
- **Target**: TIC 261136679 (18,257 data points)
- **Grid Specs**: Period 0.5–20.0 days, 5 Durations, 200–500 Phase bins

### Reproduction Commands
```bash
# Standard Resolution
astrotransit-gpu compare --preset standard

# High Resolution
astrotransit-gpu compare --preset large

# Stress Test
astrotransit-gpu compare --preset extreme
```

---

## 3. CLI Reference

### `check`
Diagnose CUDA environment and GPU hardware features.

### `compare`
Direct comparison between CPU and GPU results.
- `--preset`: Choose search scale [`standard`, `large`, `extreme`].
- `--out`: Path to save the Markdown report.

### `inject`
Perform injection/recovery experiments to evaluate detection limits.
- `--periods`: Comma-separated list of periods to inject.
- `--depths`: Comma-separated list of transit depths to inject.

### `benchmark`
Run fully reproducible experiments from YAML config files. Generates Markdown reports and plots (Periodograms, Folded Light Curves).
- `--config`: [Required] Path to YAML configuration file.

### `search`
Quick search for a single TIC target.
- `--target`: [Required] Target TIC ID.
- `--precision`: Computation precision [`float32`, `float64`].

### `batch`
Mass analysis based on a target CSV list.

---
*AstroTransit-GPU: Scaling Exoplanet Discovery with Reliability.*
