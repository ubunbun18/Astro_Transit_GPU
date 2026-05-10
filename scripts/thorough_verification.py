import numpy as np
import cupy as cp
import time
import sys
import os
import logging
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from astrotransit_gpu.search.vbls import run_vbls_massive
from astrotransit_gpu.search.cpu_reference_bls import run_astropy_bls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_lc(n_data=1312, period=3.5, duration=0.1, depth=0.01, noise=0.001, t_start=2457000.0, gaps=False):
    """合成ライトカーブを生成する"""
    time = np.linspace(t_start, t_start + 27.0, n_data)
    flux = np.zeros_like(time)
    
    # トランジットモデル (Box)
    phase = (time - t_start) % period
    in_transit = (phase < duration/2) | (phase > (period - duration/2))
    flux[in_transit] -= depth
    
    # ノイズ
    flux += np.random.normal(0, noise, size=n_data)
    
    # 重み (1/sigma^2)
    weights = np.ones_like(time) * (1.0 / (noise**2))
    
    if gaps:
        # ランダムな欠損 (50%)
        mask = np.random.rand(n_data) > 0.5
        weights[mask] = 0
        flux[mask] = 0 # 欠損値は0埋め
        
    return time, flux, weights

def test_scientific_accuracy():
    logger.info("--- Starting Scientific Accuracy Test (Float32 Precision) ---")
    
    # テストケースの設定
    true_period = 3.5214
    true_duration = 0.109375 
    true_depth = 0.01
    noise = 0.005 
    
    time_array, flux, weights = generate_synthetic_lc(period=true_period, duration=true_duration, depth=true_depth, noise=noise, t_start=0.0, gaps=False)
    
    # ゼロ平均化 (重み付き)
    w_mean = np.sum(flux * weights) / np.sum(weights)
    flux -= w_mean
    
    # 1. Astropy BLS (Reference)
    dy = np.ones_like(weights) * noise
    periods = np.linspace(3.5, 3.55, 1000)
    durations = np.array([true_duration])
    
    res_cpu = run_astropy_bls(time_array, flux, dy=dy, periods=periods, durations=durations)
    
    # 2. GPU V39 (float32)
    flux_matrix = flux.reshape(1, -1).astype(np.float32)
    weights_matrix = weights.reshape(1, -1).astype(np.float32)
    
    res_gpu_batch = run_vbls_massive(
        time_array.astype(np.float32), 
        flux_matrix, 
        cp.asarray(periods, dtype=cp.float32), 
        durations.astype(np.float32), 
        weights_matrix=cp.asarray(weights_matrix),
        dtype=np.float32
    )
    
    # 結果の抽出
    gpu_period = float(res_gpu_batch["best_period"][0].get())
    gpu_snr = float(res_gpu_batch["snr"][0].get())
    gpu_depth = float(res_gpu_batch["best_depth"][0].get())
    
    logger.info(f"Results Comparison (float32):")
    logger.info(f"Period: Astropy={res_cpu['period']:.6f}, GPU={gpu_period:.6f} (Diff={abs(res_cpu['period']-gpu_period):.2e})")
    logger.info(f"SNR:    Astropy={res_cpu['snr']:.2f}, GPU={gpu_snr:.2f} (RelErr={abs(res_cpu['snr']-gpu_snr)/res_cpu['snr']:.2%})")
    logger.info(f"Depth:  Astropy={res_cpu['depth']:.6f}, GPU={gpu_depth:.6f}")
    
    # 判定
    assert abs(res_cpu['period'] - gpu_period) < 0.01, f"Period mismatch: {abs(res_cpu['period'] - gpu_period)}"
    assert abs(res_cpu['snr'] - gpu_snr) / res_cpu['snr'] < 0.1, f"SNR mismatch: {abs(res_cpu['snr'] - gpu_snr) / res_cpu['snr']}"
    
    logger.info("Scientific Accuracy Test: PASSED")

def test_performance():
    logger.info("--- Starting Performance Benchmark ---")
    
    n_targets = 2000
    n_periods = 100000
    n_data = 1312
    
    periods = np.linspace(0.5, 13.0, n_periods)
    durations = np.linspace(0.02, 0.2, 8)
    
    # ダミーデータの生成
    flux_matrix = np.random.normal(0, 0.001, size=(n_targets, n_data)).astype(np.float32)
    time_array = np.linspace(0, 27, n_data).astype(np.float32)
    
    logger.info(f"Benchmarking with {n_targets} targets and {n_periods} periods...")
    
    start_time = time.time()
    results = run_vbls_massive(time_array, flux_matrix, cp.asarray(periods), durations)
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = n_targets / total_time
    total_searches = n_targets * n_periods
    gcps = (total_searches) / total_time / 1e9
    
    logger.info(f"Benchmark Results:")
    logger.info(f"Total Time: {total_time:.2f} s")
    logger.info(f"Throughput: {throughput:.2f} LC/s")
    logger.info(f"Giga-searches/s: {gcps:.2f} GCPS")
    
    if throughput > 300:
        logger.info("Performance Test: PASSED (Exceeded 300 LC/s target)")
    else:
        logger.warning(f"Performance Test: WARNING (Throughput {throughput:.2f} LC/s below 300 LC/s target)")

if __name__ == "__main__":
    try:
        test_scientific_accuracy()
        test_performance()
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
