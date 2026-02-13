import os
import numpy as np
from tqdm import tqdm

# Enhancement
from src.enhancement.grayscale import convert_to_grayscale
from src.enhancement.clahe import apply_clahe
from src.enhancement.noise_removal import median_filter
from src.enhancement.sharpening import sharpen_image

# Segmentation
from src.segmentation.crack_mask import generate_crack_mask

# Feature Extraction
from src.feature_extraction.length import compute_crack_length
from src.feature_extraction.width import compute_crack_width
from src.feature_extraction.density import compute_crack_density
from src.feature_extraction.orientation import compute_crack_orientation
from src.feature_extraction.texture_glcm import compute_glcm_features
from src.feature_extraction.shape_features import compute_shape_features

# Utils
from src.utils.image_loader import load_image


def extract_features_from_image(image_path):
    image = load_image(image_path)
    
    gray = convert_to_grayscale(image)
    enhanced = apply_clahe(gray)
    denoised = median_filter(enhanced)
    sharpened = sharpen_image(denoised)

    crack_mask = generate_crack_mask(sharpened)

    features = [
        compute_crack_length(crack_mask),
        compute_crack_width(crack_mask),
        compute_crack_density(crack_mask),
        compute_crack_orientation(crack_mask)
    ]

    texture = compute_glcm_features(sharpened)
    shape = compute_shape_features(crack_mask)

    features.extend([
        texture["contrast"],
        texture["homogeneity"],
        texture["energy"],
        shape["area"],
        shape["perimeter"],
        shape["aspect_ratio"]
    ])

    return features


def build_dataset(crack_dir, non_crack_dir):
    X, y = [], []

    for img in tqdm(os.listdir(crack_dir), desc="Processing crack images"):
        path = os.path.join(crack_dir, img)
        X.append(extract_features_from_image(path))
        y.append(1)  # crack

    for img in tqdm(os.listdir(non_crack_dir), desc="Processing non-crack images"):
        path = os.path.join(non_crack_dir, img)
        X.append(extract_features_from_image(path))
        y.append(0)  # non-crack

    return np.array(X), np.array(y)


if __name__ == "__main__":
    crack_dir = "dataset/SDNET2018/crack"
    non_crack_dir = "dataset/SDNET2018/non_crack"

    X, y = build_dataset(crack_dir, non_crack_dir)

    np.save("experiments/feature_vectors/X.npy", X)
    np.save("experiments/feature_vectors/y.npy", y)

    print("Dataset built successfully!")
