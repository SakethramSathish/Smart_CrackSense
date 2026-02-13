import cv2
import numpy as np

from .thresholding import adaptive_threshold
from .edge_detection import canny_edge_detection
from .morphology import morphological_opening, morphological_closing


def generate_crack_mask(
    enhanced_image: np.ndarray
) -> np.ndarray:
    """
    Generates the final binary crack mask by combining
    thresholding, edge detection, and morphological operations.

    Parameters:
    -----------
    enhanced_image : ndarray
        Enhanced grayscale image after sharpening.

    Returns:
    --------
    crack_mask : ndarray
        Final binary crack mask.
    """

    if enhanced_image is None:
        raise ValueError("Input image is None.")

    # Step 1: Adaptive Thresholding
    thresh_img = adaptive_threshold(enhanced_image)

    # Step 2: Edge Detection
    edges = canny_edge_detection(enhanced_image)

    # Step 3: Combine thresholding and edges
    combined = cv2.bitwise_or(thresh_img, edges)

    # Step 4: Morphological refinement
    opened = morphological_opening(combined)
    crack_mask = morphological_closing(opened)

    return crack_mask
