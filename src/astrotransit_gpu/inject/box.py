import numpy as np

def inject_box_transit(time, flux, period, t0, duration, depth):
    """
    Inject a simple box-shaped transit into the flux.
    
    Args:
        time (np.ndarray): Time array.
        flux (np.ndarray): Flux array.
        period (float): Period in days.
        t0 (float): Mid-transit time of the first transit.
        duration (float): Duration in days.
        depth (float): Relative depth (e.g., 0.01 for 1% depth).
        
    Returns:
        np.ndarray: Flux with injected transit.
    """
    new_flux = flux.copy()
    
    # Calculate phase
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    
    # Identify in-transit points
    in_transit = np.abs(phase) < 0.5 * duration
    
    # Apply depth
    new_flux[in_transit] *= (1.0 - depth)
    
    return new_flux
