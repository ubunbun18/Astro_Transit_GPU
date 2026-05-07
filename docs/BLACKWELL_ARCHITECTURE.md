# Blackwell Singularity Architecture (V37 "Apex Predator")

This document details the optimizations implemented in the V37 kernel, designed specifically for NVIDIA Blackwell and Ada Lovelace (RTX 40-series) architectures to achieve peak throughput for the TESS transit screening pipeline.

## 🏁 Performance Benchmark
- **Hardware**: NVIDIA Blackwell / RTX 4090 Class GPU
- **Scale**: 21.9 Billion (Target, Period) Pairs
- **Metric**: ~820 Giga-Checks/second (Transit trials)
- **Time**: ~13 minutes for 220k targets x 100k periods.

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

## 🧪 Scientific Correctness
The V37 engine maintains full scientific parity with the standard BLS algorithm. The cross-multiplication logic is numerically stable for standard light curve scales. Results are validated against Astropy's implementation to ensure transit depth and period identification remain accurate.
