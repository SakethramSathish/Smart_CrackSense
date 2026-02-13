import cv2
import numpy as np


def canny_edge_detection(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150
) -> np.ndarray:
    """
    Applies Canny edge detection to highlight crack edges.

    Parameters:
    -----------
    image : ndarray
        Input grayscale or enhanced image.
    low_threshold : int
        Lower bound for hysteresis thresholding.
    high_threshold : int
        Upper bound for hysteresis thresholding.

    Returns:
    --------
    edges : ndarray
        Binary image containing detected edges.
    """

    if image is None:
        raise ValueError("Input image is None.")

    edges = cv2.Canny(
        image,
        threshold1=low_threshold,
        threshold2=high_threshold
    )

    return edges
