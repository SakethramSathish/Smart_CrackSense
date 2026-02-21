import cv2
import numpy as np

from .thresholding import adaptive_threshold
from .edge_detection import canny_edge_detection
from .morphology import morphological_opening, morphological_closing


def generate_crack_mask(enhanced_image):

    if enhanced_image is None:
        raise ValueError("Input image is None.")

    # Stronger thresholding
    thresh_img = adaptive_threshold(
        enhanced_image,
        block_size=21,
        C=7
    )

    # Morphological cleaning
    opened = morphological_opening(thresh_img, kernel_size=5)
    closed = morphological_closing(opened, kernel_size=7)

    return closed