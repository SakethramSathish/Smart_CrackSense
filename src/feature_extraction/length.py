import numpy as np
from skimage.morphology import skeletonize


def compute_crack_length(crack_mask: np.ndarray) -> float:
    """
    Computes crack length using skeletonization and pixel count.

    Parameters:
    -----------
    crack_mask : ndarray
        Binary crack mask (0 or 255).

    Returns:
    --------
    length : float
        Estimated crack length in pixels.
    """

    if crack_mask is None:
        raise ValueError("Input crack mask is None.")

    # Convert to binary (0,1)
    binary_mask = crack_mask > 0

    # Skeletonize the crack mask
    skeleton = skeletonize(binary_mask)

    # Crack length = number of skeleton pixels
    length = np.sum(skeleton)

    return float(length)
