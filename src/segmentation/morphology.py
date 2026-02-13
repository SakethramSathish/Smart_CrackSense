import cv2
import numpy as np


def morphological_opening(
    binary_image: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Applies morphological opening to remove small noise.

    Parameters:
    -----------
    binary_image : ndarray
        Input binary image.
    kernel_size : int
        Size of the structuring element.

    Returns:
    --------
    opened_image : ndarray
        Image after morphological opening.
    """

    if binary_image is None:
        raise ValueError("Input image is None.")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )

    opened_image = cv2.morphologyEx(
        binary_image, cv2.MORPH_OPEN, kernel
    )

    return opened_image


def morphological_closing(
    binary_image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Applies morphological closing to fill gaps in crack regions.

    Parameters:
    -----------
    binary_image : ndarray
        Input binary image.
    kernel_size : int
        Size of the structuring element.

    Returns:
    --------
    closed_image : ndarray
        Image after morphological closing.
    """

    if binary_image is None:
        raise ValueError("Input image is None.")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )

    closed_image = cv2.morphologyEx(
        binary_image, cv2.MORPH_CLOSE, kernel
    )

    return closed_image
