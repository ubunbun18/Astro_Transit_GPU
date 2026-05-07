# AstroTransit-GPU Performance & Usage Guide
## ~ Maximizing Search Efficiency and Validating Numerical Precision ~

AstroTransit-GPU implements a CUDA-accelerated BLS (Box Least Squares) algorithm, providing high-throughput transit discovery while maintaining numerical parity with industry-standard tools like Astropy.

---

## 1. Performance Benchmark Report

To evaluate scalability, we compared the throughput of AstroTransit-GPU against Astropy (CPU) using identical search grid parameters.

### Execution Speed Comparison
| Search Scale | Period Grid Size | CPU (Astropy) | GPU (Ours) | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | 5,000 | ~0.83s | **0.11s** | **~7.5x** |
| **Large** | 100,000 | 16.86s | **0.21s** | **~79x** |
| **Extreme** | 1,000,000 | 159.10s | **1.21s** | **~131x** |

> [!NOTE]
> Scalability: As the number of periods increases, GPU occupancy improves significantly, achieving over **130x speedup** for grids with 1M+ periods.

### Numerical Consistency Validation
Measured at the `Standard` preset (5,000 periods) to verify physical consistency.

| Metric | CPU (Astropy) | GPU (Ours) | Difference | Agreement |
| :--- | :--- | :--- | :--- | :--- |
| **Best Period** | 6.269254 d | 6.265353 d | 0.0039 d | **99.94%** |
| **Best T0** | 1325.4994 | 3.4772 * | 0.3788 d | Phase Match |

*\* GPU calculates T0 in phase space; comparison is shown modulo Period $(P)$.*

---

## 2. Measurement Environment

These results are reproducible using the following configuration:

- **AstroTransit-GPU**: v0.1.0
- **OS**: Windows 11 / Linux (CUDA enabled)
- **GPU**: NVIDIA Compute Capability 12.0 (Blackwell Architecture)
- **CPU**: x86_64 Processor
- **Python**: 3.12.0
- **CuPy**: v13.6.0
- **Target**: TIC 261136679 (18,257 data points)
- **Grid Specs**: Period 0.5–20.0 days, 5 Durations, 200 Phase bins
- **Timing**: Synchronous measurement with `cuda.Stream.synchronize()` after GPU warm-up.

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

## 3. Full CLI Reference

### `check`: Environment Diagnostics
Verify GPU availability and hardware capabilities.

### `compare`: Benchmarking & Accuracy
Compare CPU vs GPU results directly and generate a Markdown report.
- `--target`: Target ID (default: "TIC 261136679")
- `--preset`: Preset scale [`standard`, `large`, `extreme`]
- `--out`: Report output path

### `known`: Known Planet Recovery
Validate detection logic using known targets from catalogs.
- `--target`: **[Required]** Target ID
- `--true-p`: True period for comparison

### `batch`: Parallel Search
Automated high-speed search for multiple targets from NASA archives.

### `inject-run`: Injection/Recovery Test
Perform statistical evaluation of detection limits.

### `run-config`: Configuration-based Execution
Run reproducible experiments using YAML configuration files.

---
*AstroTransit-GPU: Scaling Exoplanet Discovery with Reliability.*
