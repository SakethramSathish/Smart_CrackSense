import numpy as np


def compute_crack_density(crack_mask: np.ndarray) -> float:
    """
    Computes crack density as ratio of crack pixels
    to total image pixels.

    Parameters:
    -----------
    crack_mask : ndarray
        Binary crack mask (0 or 255).

    Returns:
    --------
    density : float
        Crack density value (0 to 1).
    """

    if crack_mask is None:
        raise ValueError("Input crack mask is None.")

    total_pixels = crack_mask.size
    crack_pixels = np.count_nonzero(crack_mask)

    density = crack_pixels / total_pixels
    return float(density)
