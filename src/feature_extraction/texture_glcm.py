import numpy as np
from skimage.feature import graycomatrix, graycoprops


def compute_glcm_features(
    gray_image: np.ndarray,
    distances: list = [1],
    angles: list = [0]
) -> dict:
    """
    Computes GLCM texture features.

    Parameters:
    -----------
    gray_image : ndarray
        Grayscale image (uint8).
    distances : list
        List of pixel pair distance offsets.
    angles : list
        List of angles (in radians).

    Returns:
    --------
    features : dict
        Dictionary containing GLCM features.
    """

    if gray_image is None:
        raise ValueError("Input image is None.")

    if len(gray_image.shape) != 2:
        raise ValueError("GLCM expects a grayscale image.")

    # Ensure image is uint8
    gray_image = gray_image.astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(
        gray_image,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    features = {
        "contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "energy": float(graycoprops(glcm, "energy")[0, 0])
    }

    return features
