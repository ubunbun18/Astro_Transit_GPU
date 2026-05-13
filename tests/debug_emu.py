import numpy as np
import math

def emulate_astropy(time, flux, dy, period, durations, oversample):
    inv_p = 1.0 / period
    min_dur = np.min(durations)
    n_bins = int(math.ceil(oversample * period / min_dur))
    
    count_binc = np.zeros(n_bins, dtype=np.float64)
    flux_binc = np.zeros(n_bins, dtype=np.float64)
    
    ivar = 1.0 / dy**2
    
    for i in range(len(time)):
        ph = math.fmod(time[i] * inv_p, 1.0)
        if ph < 0:
            ph += 1.0
        bin_idx = int(ph * n_bins)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        count_binc[bin_idx] += ivar[i]
        flux_binc[bin_idx] += flux[i] * ivar[i]
        
    for i in range(1, n_bins):
        count_binc[i] += count_binc[i - 1]
        flux_binc[i] += flux_binc[i - 1]
        
    total_counts = count_binc[-1]
    total_flux = flux_binc[-1]
    
    best_score = -1.0
    
    for start_bin in range(n_bins):
        for dur in durations:
            n_dur_bins = int(round(dur * inv_p * n_bins))
            if n_dur_bins <= 0:
                n_dur_bins = 1
            if n_dur_bins >= n_bins:
                continue
                
            end_bin = start_bin + n_dur_bins - 1
            
            if end_bin < n_bins:
                c_in = count_binc[end_bin]
                f_in = flux_binc[end_bin]
                if start_bin > 0:
                    c_in -= count_binc[start_bin - 1]
                    f_in -= flux_binc[start_bin - 1]
            else:
                wrap_end = end_bin - n_bins
                c_in = count_binc[-1] + count_binc[wrap_end]
                f_in = flux_binc[-1] + flux_binc[wrap_end]
                if start_bin > 0:
                    c_in -= count_binc[start_bin - 1]
                    f_in -= flux_binc[start_bin - 1]
                    
            if 0 < c_in < total_counts:
                r = c_in / total_counts
                s = f_in - r * total_flux
                score = (s * s) / (c_in * (1.0 - r))
                if score > best_score:
                    best_score = score
                    
    return math.sqrt(best_score)

def check_emu():
    np.random.seed(42)
    time = np.linspace(0, 27, 200).astype(np.float64)
    flux = np.random.normal(0, 0.003, size=200).astype(np.float64)
    
    true_period = 3.521
    true_dur = 0.1
    ph = (time % true_period) / true_period
    flux[ph < (true_dur / true_period)] -= 0.01
    flux -= np.median(flux)
    
    dy = np.ones_like(flux) * 0.003

    durations = np.array([0.05, 0.1, 0.15]).astype(np.float64)
    
    p = 3.0
    emu_power = emulate_astropy(time, flux, dy, p, durations, 10)
    print(f"Emulated Astropy Power for p=3.0: {emu_power}")

if __name__ == "__main__":
    check_emu()
