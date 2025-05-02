# File: ACORNN/training/vertical_sampling.py
import numpy as np

def vertical_sampling_weights(n_levs,
                              decay=0.01,
                              center_idx=39,
                              sigma=5,
                              bump_amp=0.2,
                              suppress_center=65,
                              suppress_sharpness=2):
    """
    Compute vertical sampling weights for GEOS-CF levels.

    Parameters:
    ----------
    n_levs : int
        Number of vertical levels (e.g., 72 for GEOS-CF).

    decay : float
        Exponential decay rate from the surface upward. Lower values make it flatter.

    center_idx : int
        Flipped level index (0 = surface) where the Gaussian bump is centered (e.g., 39 ≈ 150 hPa).

    sigma : float
        Width of the Gaussian bump in level index units.

    bump_amp : float
        Amplitude (strength) of the Gaussian bump.

    suppress_center : int
        Flipped level index (0 = surface) above which weights should be damped (e.g., 65 ≈ 10 hPa).

    suppress_sharpness : float
        Controls sharpness of high-altitude suppression (larger = smoother transition).

    Returns:
    --------
    weights : np.ndarray of shape (n_levs,)
        Normalized vertical sampling weights.
    """
    levels = np.arange(n_levs)
    flipped = n_levs - 1 - levels  # so 0 = surface, high index = top

    exp_part = np.exp(-decay * flipped)
    bump = bump_amp * np.exp(-((flipped - center_idx) ** 2) / (2 * sigma ** 2))
    suppression = 1 / (1 + np.exp(-(suppress_center - flipped) / suppress_sharpness))

    weights = (exp_part + bump) * suppression
    return weights / np.sum(weights)

# Default config for GEOS-CF 72 levels
def geos_cf_sampling_weights():
    return vertical_sampling_weights(
        n_levs=72,
        decay=0.01,
        center_idx=35,
        sigma=5,
        bump_amp=0.2,
        suppress_center=50,
        suppress_sharpness=3
    )
