import numpy as np
import cupy as cp
from astropy.timeseries import BoxLeastSquares
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from astrotransit_gpu.search.api import BoxLeastSquaresGPU

def calculate_exact_correlation():
    # 1. データの準備
    np.random.seed(42)
    time = np.linspace(0, 27, 2000)
    flux = np.random.normal(0, 0.003, size=2000)
    
    # 真のトランジットを注入
    true_period = 3.521
    true_dur = 0.1
    ph = (time % true_period) / true_period
    flux[ph < (true_dur / true_period)] -= 0.01
    
    # ゼロ平均化
    flux -= np.mean(flux)
    
    dy = np.ones_like(flux) * 0.003

    # 2. Astropy での計算
    periods = np.linspace(1.0, 10.0, 5000)
    durations = np.array([0.1])
    
    model_cpu = BoxLeastSquares(time, flux, dy=dy)
    res_cpu = model_cpu.power(periods, durations, objective="snr")
    
    # 3. GPU での計算
    model_gpu = BoxLeastSquaresGPU(time, flux, dy=dy)
    res_gpu = model_gpu.power(periods, durations)
    
    # 相関係数の計算
    cpu_power = res_cpu.power
    gpu_power = res_gpu.power
    
    correlation = np.corrcoef(cpu_power, gpu_power)[0, 1]
    
    print(f"--- Correlation Analysis ---")
    print(f"Number of periods evaluated: {len(periods)}")
    print(f"Exact Correlation Coefficient: {correlation:.10f}")
    
    # 追加の統計量
    print(f"Astropy Max SNR: {np.max(cpu_power):.6f}")
    print(f"GPU Max SNR:     {np.max(gpu_power):.6f}")
    print(f"Mean Absolute Error in SNR: {np.mean(np.abs(cpu_power - gpu_power)):.6f}")

if __name__ == "__main__":
    calculate_exact_correlation()
