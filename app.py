import numpy as np
import cv2
import os

# Utils
from src.utils.image_loader import load_image
from src.utils.visualization import overlay_crack_mask, save_image

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

# ML
from src.models.train_ml import train_ml_model
from src.models.evaluate import load_model, predict_severity


# =========================
# CONFIG
# =========================
MODE = "INFER"        # "TRAIN" or "INFER"
MODEL_PATH = "svm_model.pkl"

# For inference only
IMAGE_PATH = "dataset/SDNET2018/crack/001-2.jpg"


# =========================
# FEATURE EXTRACTION PIPELINE
# =========================
def extract_features(image_path):
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

    return np.array(features), image, crack_mask


# =========================
# MAIN
# =========================
def main():

    # -------- TRAIN MODE --------
    if MODE == "TRAIN":
        print("🔹 TRAIN MODE: Loading precomputed dataset")

        X = np.load("experiments/feature_vectors/X.npy")
        y = np.load("experiments/feature_vectors/y.npy")

        train_ml_model(X, y, MODEL_PATH)
        print("✅ Model training completed")

    # -------- INFERENCE MODE --------
    elif MODE == "INFER":
        print("🔹 INFERENCE MODE: Processing single image")

        features, image, crack_mask = extract_features(IMAGE_PATH)

        model, scaler = load_model(MODEL_PATH)
        severity_class, confidence = predict_severity(
            features, model, scaler
        )

        severity_map = {
            0: "Non-Crack / Minor",
            1: "Crack",
            2: "Severe Crack"
        }

        print("\n===== SMART CRACKSENSE RESULT =====")
        print(f"Predicted Severity : {severity_map.get(severity_class)}")
        print(f"Confidence Score   : {confidence:.2f}")

        # Visualization
        overlay = overlay_crack_mask(image, crack_mask)
        save_image(crack_mask, "experiments/segmented_masks/crack_mask.png")
        save_image(overlay, "experiments/plots/crack_overlay.png")

        cv2.imshow("Original Image", image)
        cv2.imshow("Crack Mask", crack_mask)
        cv2.imshow("Crack Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        raise ValueError("MODE must be 'TRAIN' or 'INFER'")


if __name__ == "__main__":
    main()
