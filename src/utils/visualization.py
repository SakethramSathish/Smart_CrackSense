import cv2
import numpy as np
import os


def show_image(window_name: str, image: np.ndarray):
    """
    Displays an image in a window.

    Parameters:
    -----------
    window_name : str
        Title of the window.
    image : ndarray
        Image to display.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def overlay_crack_mask(
    original_image: np.ndarray,
    crack_mask: np.ndarray,
    color: tuple = (0, 0, 255),
    alpha: float = 0.6
) -> np.ndarray:
    """
    Overlays crack mask on original image.

    Parameters:
    -----------
    original_image : ndarray
        Original BGR image.
    crack_mask : ndarray
        Binary crack mask.
    color : tuple
        Overlay color (default: red).
    alpha : float
        Transparency factor.

    Returns:
    --------
    overlayed_image : ndarray
        Image with crack overlay.
    """

    overlay = original_image.copy()

    # Ensure mask is binary
    mask = crack_mask > 0

    overlay[mask] = color

    overlayed_image = cv2.addWeighted(
        overlay, alpha, original_image, 1 - alpha, 0
    )

    return overlayed_image


def save_image(
    image: np.ndarray,
    save_path: str
):
    """
    Saves image to disk.

    Parameters:
    -----------
    image : ndarray
        Image to save.
    save_path : str
        Path to save image.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)
