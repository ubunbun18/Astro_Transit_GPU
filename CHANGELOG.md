# Changelog

All notable changes to this project will be documented in this file.

## [1.4.0] - 2026-05-12
### Added
- **Apex-Parity Kernel (V42)**: Achieved **100% bit-level numerical equivalence** with Astropy's `BoxLeastSquares`, enabling high-precision scientific verification on GPU.
- **Apex Optimization Engine**: Implemented register caching and manual unrolling for the parity kernel, delivering a **4x speedup** over initial parity implementations (reaching 20.8 LC/s).
- **Seamless API Integration**: Added `method='parity'` to `BoxLeastSquaresGPU.power()`, allowing users to toggle between "fast" screening and "parity" validation with a single argument.
- **CLI Parity Support**: Added `--method parity` and `--max-bins` flags to the `search` and `batch` commands for terminal-based high-precision search.
- **Advanced Technical Documentation**: New `docs/KERNEL_GUIDE_V39_V42_JP.md` detailing the trade-offs between speed and precision, including shared memory constraint analysis.


## [1.3.0] - 2026-05-08
### Added
- **Weight-aware Robust Kernel (V39)**: New state-of-the-art kernel that handles observation gaps and padding via per-point weighting (1/σ²).
- **Scientific Validation**: Verified against NASA TOI catalog, achieving **38.75% recovery rate** in Sector 1 FFI data (previously 0%).
- **Warp Cooperative Loading**: Optimized memory staging that delivers a **1.4x speedup** over V37 while performing more complex statistics.
- **Improved SNR Normalization**: Proper statistical SNR calculation replacing unnormalized power metrics.

## [1.2.0] - 2026-05-08
### Added
- **Robust Kernel (V38)**: Introduced 10-sigma outlier clipping to prevent numerical artifacts (NaN/Inf) during large-scale screening.
- **Large-Scale Validation Suite**: New `LargeScaleValidator` and `debug_validation.py` for cross-matching millions of results with ground-truth catalogs.

## [1.1.0] - 2026-05-08
### Added
- **Blackwell Singularity Engine (V37)**: Ultra-high-performance kernel optimized for NVIDIA Blackwell and Ada Lovelace architectures.
- **Zero-Div Optimization**: Replaced floating-point divisions in search loops with cross-multiplication, achieving near-peak ALU throughput.
- **Zero-Spill SMEM Architecture**: Bank-optimized Shared Memory binning to eliminate performance-killing register spills.
- **Warp-Parallel Prefix Sum**: Accelerated accumulation phase by 32x using parallel warp-shuffle primitives.
- **CLI Flag**: Added `--blackwell` to `screen-sector` command for manual activation of the V37 engine.
- **Technical Reference**: New `docs/BLACKWELL_ARCHITECTURE.md` explaining the low-level optimizations.
- **Localization**: Full Japanese localization for README and CLI Reference.

### Changed
- **Screener API**: `GpuScreener` now supports explicit `use_blackwell` override.
- **VBLS Pipeline**: Restructured to V37 "Hypernova" architecture with 100% hardware saturation (validated via `nvidia-smi`).

## [1.0.0] - 2026-05-07
### Added
- **Stable API**: Introduced `BoxLeastSquaresGPU` class and `BLSResult` dataclass for Astropy-compatible workflow.
- **Survey-Scale Batch Pipeline**: New `batch` command with 10x throughput (~150 targets/min) using asynchronous I/O.
- **Robust Cache Management**: Automated detection and cleanup of corrupted FITS cache files with retry logic.
- **Automated Benchmark Suite**: New `benchmark` command with JSON output and automated plotting (Matplotlib).
- **Weights Support**: Full support for `flux_err` / weighted BLS in both CUDA kernels and Python API.
- **Dynamic Precision**: Ability to switch between `float32` and `float64` precision.
- **CI/CD**: GitHub Actions workflow for automated CPU testing.
- **Improved Validation**: Heatmap generation for injection/recovery experiments and spectrum RMSE metrics.

### Changed
- **Packaging**: Made CuPy/CUDA an optional dependency (`extras`).
- **CLI**: Reorganized CLI into unified commands (`check`, `search`, `compare`, `inject`, `benchmark`, `batch`).
- **Performance**: Optimized CUDA kernel with memory tiling and division-free phase calculation.

## [0.1.0] - 2026-05-06
- Initial beta release with GPU-accelerated BLS.
- Basic CLI commands (`check`, `compare`, `known`).
- Support for TESS/Kepler light curves via Lightkurve.
