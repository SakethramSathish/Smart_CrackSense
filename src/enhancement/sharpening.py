import cv2
import numpy as np


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """
    Sharpens an image using a Laplacian-based kernel.

    Parameters:
    -----------
    image : ndarray
        Input grayscale image.

    Returns:
    --------
    sharpened_image : ndarray
        Sharpened image.
    """

    if image is None:
        raise ValueError("Input image is None.")

    # Sharpening kernel
    kernel = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ])

    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image
