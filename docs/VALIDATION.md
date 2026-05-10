# Scientific Validation Protocol

AstroTransit-GPU prioritizes both high throughput and scientific integrity. This document defines the formal procedures and criteria for validating transit detection performance.

## 1. Objectives
To quantitatively and reproducibly evaluate the impact of GPU kernel updates (e.g., V39) on the recovery rate (Completeness) and period identification accuracy (Accuracy) of known planets.

## 2. Standard Criteria
To ensure reproducibility, the following criteria are fixed in `configs/validation_v39.yaml`:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **SNR Threshold** | 7.1 | TESS standard significance level. |
| **Period Tolerance (p_tol)** | 0.02 (2%) | Relative error between $P_{det}$ and $P_{true}$ must be within 2%. |
| **T0 Tolerance (t0_tol)** | 0.5 days | Maximum allowed shift in transit midpoint. |
| **Require T0 Match** | True | Require both period and phase alignment for a successful match. |

## 3. Validation Procedure

### 3.1 Prerequisites
Validation requires the results CSV from a full-sector screening (~25MB).
```bash
# Run full screening (if not already done)
python -m astrotransit_gpu screen-sector --blackwell --out outputs/bench_219331_v39.csv
```

### 3.2 Running the Validation Script
Generate the statistical report using the standard configuration:
```bash
python scripts/reproduce_v39_validation.py --results outputs/bench_219331_v39.csv
```

## 4. Metric Definitions

### 4.1 Completeness
The ratio of recovered TOIs to physically detectable TOIs (those with at least 2 transits within the observation baseline).
$$Completeness = \frac{Recovered\ TOIs}{Detectable\ TOIs}$$

> [!NOTE]
> **Notes on Recovery Metrics**:
> - **38.75%**: Result from optimized small-scale tests focusing on Sector 1 targets only.
> - **17.31%**: Result from massive all-sky screening (219,331 targets) using standard V39 validation criteria (`require_t0: true`).

### 4.2 False Positive Rate (FPR)
The ratio of significant signals (SNR > 7.1) that do not match known TOIs or Eclipsing Binary (EB) catalogs.
*Note: This currently includes all unvetted candidates, hence the high initial values.*

## 5. Catalog Sources
To ensure consistent results, the following sources are used:
- **TOI Catalog**: NASA Exoplanet Archive (Auto-fetched)
- **EB Catalog**: `data/catalogs/eb_latest.csv` (Included in repo)

If online access fails, local caches in `data/catalogs/` are automatically used as fallbacks.
