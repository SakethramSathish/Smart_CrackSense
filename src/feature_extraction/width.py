import numpy as np
import cv2
from skimage.morphology import skeletonize


def compute_crack_width(crack_mask: np.ndarray) -> float:
    """
    Computes average crack width using distance transform.

    Parameters:
    -----------
    crack_mask : ndarray
        Binary crack mask (0 or 255).

    Returns:
    --------
    avg_width : float
        Estimated average crack width in pixels.
    """

    if crack_mask is None:
        raise ValueError("Input crack mask is None.")

    # Convert to binary
    binary_mask = (crack_mask > 0).astype(np.uint8)

    # Compute distance transform
    distance_map = cv2.distanceTransform(
        binary_mask, cv2.DIST_L2, 5
    )

    # Skeletonize to get centerline
    skeleton = skeletonize(binary_mask > 0)

    # Width at each skeleton pixel = 2 * distance
    widths = distance_map[skeleton] * 2

    if len(widths) == 0:
        return 0.0

    avg_width = np.mean(widths)
    return float(avg_width)
