# Changelog

All notable changes to this project will be documented in this file.

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
