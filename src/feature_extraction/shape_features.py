import cv2
import numpy as np


def compute_shape_features(crack_mask: np.ndarray) -> dict:
    """
    Computes shape-based features from crack mask.

    Parameters:
    -----------
    crack_mask : ndarray
        Binary crack mask (0 or 255).

    Returns:
    --------
    features : dict
        Dictionary containing shape features.
    """

    if crack_mask is None:
        raise ValueError("Input crack mask is None.")

    # Ensure binary format
    binary = (crack_mask > 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "aspect_ratio": 0.0
        }

    # Use largest contour (main crack)
    largest_contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, closed=True)

    # Bounding rectangle for aspect ratio
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h != 0 else 0.0

    features = {
        "area": float(area),
        "perimeter": float(perimeter),
        "aspect_ratio": float(aspect_ratio)
    }

    return features
