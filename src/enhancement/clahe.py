import cv2
import numpy as np


def apply_clahe(
    gray_image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to a grayscale image.

    Parameters:
    -----------
    gray_image : ndarray
        Input grayscale image.
    clip_limit : float
        Threshold for contrast limiting.
    tile_grid_size : tuple
        Size of grid for histogram equalization.

    Returns:
    --------
    enhanced_image : ndarray
        Contrast enhanced image.
    """

    if gray_image is None:
        raise ValueError("Input image is None.")

    if len(gray_image.shape) != 2:
        raise ValueError("CLAHE expects a grayscale image.")

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced_image = clahe.apply(gray_image)
    return enhanced_image
