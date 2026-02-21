import cv2
import numpy as np
import math


def compute_shape_features(crack_mask: np.ndarray):

    if crack_mask is None:
        raise ValueError("Input crack mask is None.")

    binary = (crack_mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    MIN_AREA = 100
    MIN_ELONGATION = 2.5  # 🔥 new filter

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)

        if area == 0:
            continue

        elongation = (perimeter ** 2) / (4 * math.pi * area)

        if elongation > MIN_ELONGATION:
            valid_contours.append(contour)

    if len(valid_contours) == 0:
        return {
            "num_cracks": 0,
            "total_area": 0.0,
            "avg_area": 0.0,
            "total_perimeter": 0.0,
            "avg_aspect_ratio": 0.0
        }

    areas = []
    perimeters = []
    aspect_ratios = []

    for contour in valid_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0

        areas.append(area)
        perimeters.append(perimeter)
        aspect_ratios.append(aspect_ratio)

    return {
        "num_cracks": len(valid_contours),
        "total_area": float(np.sum(areas)),
        "avg_area": float(np.mean(areas)),
        "total_perimeter": float(np.sum(perimeters)),
        "avg_aspect_ratio": float(np.mean(aspect_ratios))
    }