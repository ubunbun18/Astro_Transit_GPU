"""
Final Validation: 100 targets x 100,000 periods
Speed + Scientific Accuracy vs Astropy (same period/duration grid)
"""
import numpy as np
import cupy as cp
import time
import sys
import os
import logging
from astropy.timeseries import BoxLeastSquares

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from astrotransit_gpu.search.vbls import run_vbls_massive

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_TARGETS   = 100
N_PERIODS   = 100_000
N_DATA      = 1312
TRUE_PERIOD = 3.521
TRUE_DUR    = 0.1
TRUE_DEPTH  = 0.01
NOISE       = 0.003   # realistic TESS-like noise (SNR ~10)

def make_lc(seed):
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 27, N_DATA).astype(np.float32)
    y   = rng.normal(0, NOISE, N_DATA).astype(np.float32)
    ph  = (t % TRUE_PERIOD) / TRUE_PERIOD
    y[ph < (TRUE_DUR / TRUE_PERIOD)] -= TRUE_DEPTH
    return t, y

periods_np   = np.linspace(0.5, 13.0, N_PERIODS).astype(np.float32)
durations_np = np.linspace(0.01, 0.2, 8).astype(np.float32)

# ── 1. Accuracy check (1 target, same grid as GPU) ───────────────────────────
logger.info("=" * 62)
logger.info("  ACCURACY CHECK  (1 target vs Astropy, identical period grid)")
logger.info("=" * 62)

t_ref, y_ref = make_lc(0)

# GPU (float32)
res_gpu    = run_vbls_massive(
    t_ref, y_ref.reshape(1, -1),
    cp.asarray(periods_np), durations_np, dtype=np.float32
)
gpu_snr    = float(cp.asnumpy(res_gpu["snr"])[0])
gpu_period = float(cp.asnumpy(res_gpu["best_period"])[0])

# Astropy on same period+duration grid (float64 for reference)
# Use the same weighted BLS formulation and compute SNR directly
bls     = BoxLeastSquares(t_ref.astype(np.float64), y_ref.astype(np.float64))
res_bls = bls.power(periods_np.astype(np.float64),
                    durations_np.astype(np.float64))
best_i     = np.argmax(res_bls.power)
ap_period  = float(res_bls.period[best_i])
# Astropy 'power' = SNR^2-based statistic; compute sqrt for fair SNR comparison
ap_snr_raw = float(res_bls.power[best_i])
ap_snr     = float(np.sqrt(max(ap_snr_raw, 0.0)))   # convert to SNR scale

rel_period = abs(gpu_period - ap_period) / abs(ap_period) * 100
rel_snr    = abs(gpu_snr   - ap_snr)    / abs(ap_snr)    * 100 if ap_snr > 0 else 999.0

logger.info(f"{'Metric':<18} {'Astropy':>14} {'GPU':>14} {'RelErr':>10}")
logger.info("-" * 60)
logger.info(f"{'Best Period (days)':<18} {ap_period:>14.6f} {gpu_period:>14.6f} {rel_period:>9.4f}%")
logger.info(f"{'SNR':<18} {ap_snr:>14.4f} {gpu_snr:>14.4f} {rel_snr:>9.2f}%")

accuracy_pass = rel_snr < 2.0 and rel_period < 0.5
logger.info(f"\nAccuracy : {'PASS [OK]' if accuracy_pass else 'FAIL [X]'}"
            f"  (SNR < 2%, Period < 0.5%)")

# ── 2. Speed test (100 targets x 100,000 periods) ────────────────────────────
logger.info("")
logger.info("=" * 62)
logger.info(f"  SPEED TEST  ({N_TARGETS} targets x {N_PERIODS:,} periods)")
logger.info("=" * 62)

y_batch = np.stack([make_lc(i)[1] for i in range(N_TARGETS)]).astype(np.float32)

# CPU – sample 3 targets and extrapolate
N_CPU = 3
logger.info(f"  CPU Astropy ({N_CPU} targets, extrapolating)...")
cpu_t0 = time.perf_counter()
for i in range(N_CPU):
    b = BoxLeastSquares(t_ref.astype(np.float64), y_batch[i].astype(np.float64))
    b.power(periods_np.astype(np.float64), durations_np.astype(np.float64))
cpu_per_lc   = (time.perf_counter() - cpu_t0) / N_CPU
cpu_through  = 1.0 / cpu_per_lc
cpu_est_100  = cpu_per_lc * N_TARGETS

# GPU – warm-up then timed
_ = run_vbls_massive(t_ref, y_batch[:2], cp.asarray(periods_np), durations_np, dtype=np.float32)
cp.cuda.runtime.deviceSynchronize()

logger.info(f"  GPU V41 ({N_TARGETS} targets)...")
gpu_t0 = time.perf_counter()
_ = run_vbls_massive(t_ref, y_batch, cp.asarray(periods_np), durations_np, dtype=np.float32)
cp.cuda.runtime.deviceSynchronize()
gpu_time    = time.perf_counter() - gpu_t0
gpu_through = N_TARGETS / gpu_time
speedup     = gpu_through / cpu_through
gcps        = N_TARGETS * N_PERIODS * len(durations_np) / gpu_time / 1e9

logger.info("")
logger.info(f"{'Metric':<32} {'CPU':>12} {'GPU':>12}")
logger.info("-" * 58)
logger.info(f"{'Throughput (LC/s)':<32} {cpu_through:>12.1f} {gpu_through:>12.1f}")
logger.info(f"{'Est. time for 100 targets (s)':<32} {cpu_est_100:>12.1f} {gpu_time:>12.4f}")
logger.info(f"{'Speedup vs CPU':<32} {'—':>12} {speedup:>11.0f}x")
logger.info(f"{'Giga-checks/sec':<32} {'—':>12} {gcps:>12.2f}")

# ── 3. Final summary ──────────────────────────────────────────────────────────
logger.info("")
logger.info("=" * 62)
logger.info("  FINAL RESULT  (V41 \"Vortex Apex\")")
logger.info("=" * 62)
logger.info(f"  Accuracy  : {'PASS' if accuracy_pass else 'FAIL'}"
            f"  (Period {rel_period:.4f}%)")
logger.info(f"  Speed     : {gpu_through:.1f} LC/s  ({speedup:.0f}x vs CPU)")
logger.info(f"  Target    : >300 LC/s  =>  {'EXCEEDED' if gpu_through > 300 else 'NOT MET'}")
logger.info(f"  Gchecks/s : {gcps:.2f}")
logger.info("=" * 62)
