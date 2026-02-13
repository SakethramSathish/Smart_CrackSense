import pickle
import numpy as np


def load_model(model_path: str):
    """
    Loads trained ML model and scaler.

    Parameters:
    -----------
    model_path : str
        Path to saved model file.

    Returns:
    --------
    model : trained classifier
    scaler : feature scaler
    """

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    return data["model"], data["scaler"]


def predict_severity(
    features: np.ndarray,
    model,
    scaler
):
    """
    Predicts crack severity and confidence score.

    Parameters:
    -----------
    features : ndarray
        Feature vector for one sample.
    model : trained classifier
    scaler : fitted scaler

    Returns:
    --------
    predicted_class : int
        Predicted severity class.
    confidence : float
        Confidence score of prediction.
    """

    # Scale features
    features_scaled = scaler.transform([features])

    # Predict class
    predicted_class = model.predict(features_scaled)[0]

    # Predict confidence
    confidence = np.max(
        model.predict_proba(features_scaled)
    )

    return predicted_class, confidence
