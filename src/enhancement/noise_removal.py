import cv2
import numpy as np


def median_filter(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Applies median filtering to remove salt-and-pepper noise.

    Parameters:
    -----------
    image : ndarray
        Input grayscale image.
    kernel_size : int
        Size of the median filter kernel (must be odd).

    Returns:
    --------
    filtered_image : ndarray
        Noise reduced image.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: int = 75,
    sigma_space: int = 75
) -> np.ndarray:
    """
    Applies bilateral filtering to reduce noise while
    preserving edges.

    Parameters:
    -----------
    image : ndarray
        Input grayscale image.
    d : int
        Diameter of pixel neighborhood.
    sigma_color : int
        Filter sigma in color space.
    sigma_space : int
        Filter sigma in coordinate space.

    Returns:
    --------
    filtered_image : ndarray
        Edge-preserving filtered image.
    """

    if image is None:
        raise ValueError("Input image is None.")

    filtered_image = cv2.bilateralFilter(
        image, d, sigma_color, sigma_space
    )
    return filtered_image
