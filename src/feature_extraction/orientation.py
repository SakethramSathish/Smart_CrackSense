import numpy as np
import cv2


def compute_crack_orientation(crack_mask: np.ndarray) -> float:
    """
    Computes dominant crack orientation using Hough Line Transform.

    Parameters:
    -----------
    crack_mask : ndarray
        Binary crack mask (0 or 255).

    Returns:
    --------
    dominant_angle : float
        Dominant crack orientation in degrees.
        Returns -1 if no dominant orientation is found.
    """

    if crack_mask is None:
        raise ValueError("Input crack mask is None.")

    # Edge detection on crack mask
    edges = cv2.Canny(crack_mask, 50, 150)

    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None:
        return -1.0

    # Extract angles (theta)
    angles = [line[0][1] for line in lines]

    # Convert radians to degrees
    angles_deg = [angle * 180 / np.pi for angle in angles]

    # Compute dominant orientation (mean angle)
    dominant_angle = np.mean(angles_deg)

    return float(dominant_angle)
