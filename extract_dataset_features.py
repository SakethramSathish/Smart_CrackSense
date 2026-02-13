import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.image_loader import load_image
from src.enhancement.grayscale import convert_to_grayscale
from src.enhancement.clahe import apply_clahe
from src.enhancement.noise_removal import median_filter
from src.enhancement.sharpening import sharpen_image
from src.segmentation.crack_mask import generate_crack_mask
from src.feature_extraction.length import compute_crack_length
from src.feature_extraction.width import compute_crack_width
from src.feature_extraction.density import compute_crack_density
from src.feature_extraction.orientation import compute_crack_orientation
from src.feature_extraction.texture_glcm import compute_glcm_features
from src.feature_extraction.shape_features import compute_shape_features


def extract_features_from_image(image_path):
    """Extract features from a single image."""
    try:
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

        return np.array(features)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def extract_dataset_features():
    """Extract features from all dataset images."""
    dataset_path = "dataset/SDNET2018"
    X = []
    y = []

    # Process crack images (label: 1)
    print("Processing crack images...")
    crack_dir = os.path.join(dataset_path, "crack")
    if os.path.exists(crack_dir):
        crack_files = [f for f in os.listdir(crack_dir) if f.endswith(('.jpg', '.png'))]
        for i, filename in enumerate(crack_files):
            print(f"  [{i+1}/{len(crack_files)}] {filename}")
            features = extract_features_from_image(os.path.join(crack_dir, filename))
            if features is not None:
                X.append(features)
                y.append(1)  # Crack

    # Process non-crack images (label: 0)
    print("\nProcessing non-crack images...")
    non_crack_dir = os.path.join(dataset_path, "non_crack")
    if os.path.exists(non_crack_dir):
        non_crack_files = [f for f in os.listdir(non_crack_dir) if f.endswith(('.jpg', '.png'))]
        for i, filename in enumerate(non_crack_files):
            print(f"  [{i+1}/{len(non_crack_files)}] {filename}")
            features = extract_features_from_image(os.path.join(non_crack_dir, filename))
            if features is not None:
                X.append(features)
                y.append(0)  # Non-crack

    if len(X) == 0:
        print("No features extracted! Check dataset paths.")
        return

    # Save feature vectors
    X = np.array(X)
    y = np.array(y)

    os.makedirs("experiments/feature_vectors", exist_ok=True)
    np.save("experiments/feature_vectors/X.npy", X)
    np.save("experiments/feature_vectors/y.npy", y)

    print(f"\n✅ Features extracted successfully!")
    print(f"   Total samples: {len(X)}")
    print(f"   Features per sample: {X.shape[1]}")
    print(f"   Crack samples: {np.sum(y == 1)}")
    print(f"   Non-crack samples: {np.sum(y == 0)}")


if __name__ == "__main__":
    extract_dataset_features()
