import cv2
import numpy as np


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 11,
    C: int = 2
) -> np.ndarray:
    """
    Applies adaptive thresholding to segment crack regions.

    Parameters:
    -----------
    image : ndarray
        Input enhanced grayscale image.
    block_size : int
        Size of pixel neighborhood (must be odd).
    C : int
        Constant subtracted from the mean.

    Returns:
    --------
    binary_image : ndarray
        Binary image highlighting crack regions.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if block_size % 2 == 0:
        raise ValueError("block_size must be odd.")

    binary_image = cv2.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=C
    )

    return binary_image
