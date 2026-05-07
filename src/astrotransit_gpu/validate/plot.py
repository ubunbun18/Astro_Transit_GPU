import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison(periods, cpu_power, gpu_power, output_path):
    """
    Generate a comparison plot of CPU vs GPU power spectra.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Power Spectra
    ax1.plot(periods, cpu_power, label="CPU (Astropy)", alpha=0.7)
    ax1.plot(periods, gpu_power, label="GPU (AstroTransit)", alpha=0.7, linestyle="--")
    ax1.set_ylabel("Power")
    ax1.set_title("BLS Power Spectrum Comparison")
    ax1.legend()
    
    # 2. Residuals
    residuals = gpu_power - cpu_power
    ax2.plot(periods, residuals, color='red', alpha=0.6)
    ax2.set_ylabel("Residual (GPU - CPU)")
    ax2.set_xlabel("Period (days)")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_folded_lc(t, y, period, t0, output_path):
    """
    Plot folded light curve at the best period.
    """
    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5
    plt.figure(figsize=(10, 4))
    plt.scatter(phase, y, s=1, color='black', alpha=0.3)
    plt.xlabel("Phase")
    plt.ylabel("Normalized Flux")
    plt.title(f"Folded Light Curve (P={period:.6f} d)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
