import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts a BGR image to grayscale.

    Parameters:
    -----------
    image : ndarray
        Input color image in BGR format.

    Returns:
    --------
    gray_image : ndarray
        Grayscale image.
    """

    if image is None:
        raise ValueError("Input image is None.")

    # If image is already grayscale, return as is
    if len(image.shape) == 2:
        return image

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
