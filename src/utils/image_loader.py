import os
import cv2
from typing import List, Tuple


def load_images_from_folder(
    folder_path: str,
    resize: Tuple[int, int] = None,
    as_gray: bool = False
) -> List:
    """
    Loads all images from a given folder.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing images.
    resize : tuple(int, int), optional
        Resize images to (width, height). Default is None.
    as_gray : bool
        If True, images are converted to grayscale.

    Returns:
    --------
    images : list
        List of loaded images.
    """

    images = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip non-image files
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
            continue

        # Load image
        image = cv2.imread(file_path)

        if image is None:
            continue

        # Convert to grayscale if required
        if as_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize image if required
        if resize is not None:
            image = cv2.resize(image, resize)

        images.append(image)

    return images


def load_image(
    image_path: str,
    resize: Tuple[int, int] = None,
    as_gray: bool = False
):
    """
    Loads a single image from a given path.

    Parameters:
    -----------
    image_path : str
        Path to the image file.
    resize : tuple(int, int), optional
        Resize image to (width, height).
    as_gray : bool
        Convert image to grayscale.

    Returns:
    --------
    image : ndarray
        Loaded image.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Failed to load image.")

    if as_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if resize is not None:
        image = cv2.resize(image, resize)

    return image
