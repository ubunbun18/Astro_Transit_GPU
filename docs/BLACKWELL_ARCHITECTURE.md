# Blackwell Singularity Architecture (V39 "Apex Predator")

This document details the optimizations implemented in the V37 and V39 kernels, designed specifically for NVIDIA Blackwell and Ada Lovelace (RTX 40-series) architectures to achieve peak throughput for the TESS transit screening pipeline.

## 🏁 Performance Benchmark
- **Hardware**: NVIDIA Blackwell / RTX 4090 Class GPU
- **Scale**: 21.9 Billion (Target, Period) Pairs
- **Metric**: ~1,100 Giga-Checks/second (V39 Effective)
- **Time**: ~9.4 minutes for 220k targets x 100k periods.

## 🛠️ Core Optimizations

### 1. Winner-Take-All Output (Screening Optimization)
Unlike the standard BLS kernel which writes the entire power spectrum (periodogram) to global memory, the **V37 engine only outputs the single best candidate per target**.
- **Reason**: Writing millions of floating-point values to global memory is a massive bandwidth bottleneck. By using block-level reductions and only performing one `atomicMax` per target, we achieve near-instantaneous results even for 1M+ period grids.
- **Limitation**: V37 cannot be used to generate periodogram plots. For detailed analysis of a specific candidate, use the standard kernel (which is automatically used by the `search` command).

### 2. Zero-Div Logic (Cross-Multiplication)
The Box Least Squares (BLS) score formula involves a division:
`score = (numer^2) / (count * (total_count - count))`

Floating-point division on NVIDIA GPUs has a latency of 10-20 cycles. By rewriting the comparison:
`cur_numer^2 / cur_denom > best_numer^2 / best_denom`
as:
`cur_numer^2 * best_denom > best_numer^2 * cur_denom`
we replace a high-latency division with two 1-cycle multiplications. This allows the search loop to run at near-theoretical ALU throughput.

### 2. Zero-Spill SMEM Accumulators
Each thread originally required 128 registers for binning. With 256+ threads per block, this caused "Register Spilling" to slow Local Memory (Lmem).
V37 moves these accumulators to **Shared Memory (SMEM)**. 
- **Bank Optimization**: Uses a `[64][129]` layout. The odd row size (129) ensures that threads in a warp access different SMEM banks simultaneously, resulting in **zero bank conflicts**.

### 3. Warp-Cooperative Parallel Scan
Binary binning requires a Prefix Sum (Cumulative Sum) of the bins. 
- **Legacy**: One thread per warp performed the sum sequentially (128 cycles).
- **V37**: Uses `__shfl_up_sync` and `__shfl_sync` to perform a parallel warp-scan. The 128 bins are processed in chunks of 32, reducing the accumulation time by 32x.

### 4. Block-Level Representative Election
To minimize global memory contention, V37 performs a block-wide reduction to find the single best transit candidate among the 64 periods processed in a block. Only one thread per block performs the final `atomicMax` update to global memory, reducing atomic contention by a factor of 64.

## 🛠️ V39 Weight-aware Evolution

V39 introduces significant enhancements for scientific robustness while further increasing throughput.

### 5. Warp-Cooperative Memory Staging (V39)
To handle weighted SNR, V39 must load `flux`, `time`, and `weight` arrays simultaneously.
- **Optimization**: Uses 512 threads per block to perform a "Cooperative Load" into SMEM. This maximizes memory bus saturation and provides a **1.4x speedup** over the V37 loading pattern by better aligning with Blackwell's L2 cache lines.

### 6. Gap-Handling Weighted SNR (V39)
V39 replaces simple point counts with a **Weighted Sum of Squares** approach:
`SNR = |total_W * S_WF - S_W * total_WF| / sqrt(S_W * (total_W - S_W) * total_W)`
where `W` is the weight ($1/\sigma^2$) and `WF` is weighted flux. This allows the kernel to **statistically ignore observation gaps (padding)**, which is critical for recovering planets in FFI data with large gaps (e.g., TESS downlink breaks).

## 🧪 Scientific Correctness
The V39 engine is the current production standard. It has been validated against the NASA TOI catalog, achieving a **17.31% recovery rate** on a large-scale subset of TESS data (219,331 targets), compared to 0% for unweighted versions that were susceptible to gap artifacts.
*Note: Recovery rate reaches 38.75% in Sector 1 focused tests. See [VALIDATION.md](./VALIDATION.md) for details.*
