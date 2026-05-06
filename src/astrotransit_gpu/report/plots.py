import matplotlib.pyplot as plt
import numpy as np
import os

def plot_periodogram(periods, power, best_period=None, output_path=None):
    """Plot the BLS power spectrum."""
    plt.figure(figsize=(10, 4))
    plt.plot(periods, power, color='black', lw=0.5)
    if best_period:
        plt.axvline(best_period, color='red', alpha=0.5, ls='--', label=f'Best: {best_period:.4f} d')
        plt.legend()
    plt.xlabel("Period (days)")
    plt.ylabel("BLS Power (SNR)")
    plt.title("Periodogram")
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def plot_folded_lightcurve(time, flux, period, t0, duration, output_path=None):
    """Plot the phase-folded light curve."""
    # Phase folding
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    
    plt.figure(figsize=(8, 5))
    plt.scatter(phase, flux, s=1, color='gray', alpha=0.5, label='Data')
    
    # Binned version for visibility
    n_bins = 50
    bin_edges = np.linspace(-0.5 * period, 0.5 * period, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_flux = []
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if np.any(mask):
            binned_flux.append(np.median(flux[mask]))
        else:
            binned_flux.append(np.nan)
            
    plt.plot(bin_centers, binned_flux, color='red', lw=2, label='Binned Median')
    
    plt.axvline(-0.5 * duration, color='blue', ls=':', alpha=0.5)
    plt.axvline(0.5 * duration, color='blue', ls=':', alpha=0.5)
    
    plt.xlabel("Time from mid-transit (days)")
    plt.ylabel("Normalized Flux")
    plt.title(f"Phase Folded (P={period:.4f} d)")
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
